# Import pytorch
import torch
import torchvision

# Import needed utils
from utils.tcolors import terminalColor as tc
from utils.pyramids import GaussianPyramids, GaborPyramids

# Main class
class IttisMethod:
    """A class for generating a saliency map using Itti's method from the Paper: 
    ``A Model of Saliency-Based Visual Attention for Rapid Scene Analysis``.
    """
    def __init__(self, c: list = [2, 3, 4], delta: list = [3, 4]) -> None:
        """Initializes the IttisMethod class.

        Args:
            c (list, optional): CENTER-surround scales. Defaults to [2, 3, 4].
            delta (list, optional): center-SURROUND scales. Defaults to [3, 4].
        """
        self.c = c
        self.delta = delta

        # scales "sigma" for pyramid images
        self.sigma = [i for i in range(0, max(self.c) + max(self.delta) + 1)]
        # orientations 
        self.theta = [0, 45, 90, 135]
    
    def __check_image_dimensions(self, img: torch.Tensor) -> None:
        """Checks if the dimensions of the input image are in the right format!

        Args:
            img (torch.Tensor): rgb input image

        Raises:
            ValueError: If shape of ``img`` is incorrect
            ValueError: if order of axes are incorrect e.g. Numpy format (H, W, C)
            ValueError: If number of color channels aren't 3
            ValueError: If shape if image is less than 256 pixels per side
            ValueError: If image is not in range [0, 1]
        """
        if len(img.shape) != 3:
            raise ValueError(tc.err + "input images must be in shape of: (C, H, W)!")
        if img.shape[0] > img.shape[1] or img.shape[0] > img.shape[2]:
            raise ValueError(tc.err + "Expect the format: (C{hannel}, H{eight}, W{idth})!")
        if img.shape[0] != 3:
            raise ValueError(tc.err + f"Only rgb color images are allowed! Got image with {img.shape[0]} layers instead of 3!")
        if img.shape[1] < 256 or img.shape[2] < 256:
            raise ValueError(tc.err + "Shape of image has to be greater than 256 x 256 pixels!")
        if torch.max(img) > 1.0:
            raise ValueError(tc.err + "Image values should be in range [0, 1]!")
    
    def colors(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the color features for the given rgb image. 

        Args:
            img (torch.Tensor): rgb input image

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: (R, G, B, Y) features
        """
        # Check if image is in right format!
        self.__check_image_dimensions(img)
        
        rc = img[..., 0, :, :]
        gc = img[..., 1, :, :]
        bc = img[..., 2, :, :]
        
        intensity_image = (rc + gc + bc) / 3.0
        
        idx = torch.where(intensity_image < (torch.max(intensity_image) / 10))
        
        r = rc / intensity_image
        g = gc / intensity_image
        b = bc / intensity_image
        
        r[idx] = 0
        g[idx] = 0
        b[idx] = 0
        
        R = r - (g + b) / 2
        G = g - (r + b) / 2
        B = b - (r + g) / 2
        Y = (r + g) / 2 - torch.abs(r - g) / 2 - b
        return R[None, ...], G[None, ...], B[None, ...], Y[None, ...]

    def intensity(self, img: torch.Tensor) -> torch.Tensor:
        """Creates an intensity map from the given rgb image.

        Args:
            img (torch.Tensor): rgb input image

        Returns:
            torch.Tensor: intensity map
        """
        r = img[..., 0, :, :]
        g = img[..., 1, :, :]
        b = img[..., 2, :, :]
        result = (r + g + b) / 3.0
        return result[None, ...]

    def orientations(self, img: torch.Tensor) -> list:
        """Generates Gabor pyramids for the given input image. 
        For that purpose the image will be convertet using the ``intensity`` function and
        then applied to the GaborPyramids class, which yield the gabor pyramids.

        Args:
            img (torch.Tensor): rgb input image

        Returns:
            list: a list of orientations of pyramids 
        """
        # Check if image is in right format!
        self.__check_image_dimensions(img)
        
        intensity = self.intensity(img)
        
        gaborp_gen = GaborPyramids(sigma=self.sigma, theta=self.theta)
        gabor_pyramids = gaborp_gen.generate(intensity)
        
        return gabor_pyramids
        
    def linear_filtering(self, img: torch.Tensor) -> tuple:
        """Performs the feature extraction and the follow up pyramid image generation. 

        Args:
            img (torch.Tensor): input rgb image

        Returns:
            tuple: Results of features
        """
        # Check if image is in right format!
        self.__check_image_dimensions(img)
        
        #######################
        #   Create features   #
        #######################
        # Intensity
        I = self.intensity(img)
        
        # colors
        R, G, B, Y = self.colors(img)
        
        # orientation
        O = self.orientations(img)
        
        # Construct pyramid generators
        gp_gen = GaussianPyramids(sigma=self.sigma)

        # Generate pyramid images for every feature except orientation (they are already pyramid images)
        ## Intensity features
        Ip = gp_gen.generate(I)
        
        ## Color features
        Rp = gp_gen.generate(R)
        Gp = gp_gen.generate(G)
        Bp = gp_gen.generate(B)
        Yp = gp_gen.generate(Y)
        
        return (Ip, O, Rp, Gp, Bp, Yp)
    
    def center_surrounded_differences(self, PyramidList_c: list, PyramidList_s) -> list:
        """Calculates the center-surrounded difference (:math:`\ominus`) for the given pyramid images.
        
        For a given pyramid image :math:`I(\sigma)` it performs the following equation for the two scales ``c`` and ``s``:
        
        .. math:: I(s,c) = | I_c(c) \ominus I_s(s) |

        Args:
            PyramidList_c (list): list of pyramid images for c scales [from the previous function ``linear_filtering(...)``]
            PyramidList_s (list): list of pyramid images for s scales [from the previous function ``linear_filtering(...)``]

        Raises: 
            ValueError: If length of input list `PyramidList_c` isn't long enough 
            ValueError: If length of input list `PyramidList_s` isn't long enough 

        Returns:
            list: csd maps
        """
        if len(PyramidList_c) < max(self.c):
            raise ValueError(tc.err + "Input list `PyramidList_c` isn't long enough for the running index!")
        if len(PyramidList_s) < (max(self.c) + max(self.delta)):
            raise ValueError(tc.err + "Input list `PyramidList_s` isn't long enough for the running index!")
        
        csd = []
        
        for c_ in self.c:
            for d in self.delta:
                # In general finer resolution
                pyramid_c = PyramidList_c[c_]
                # In general coarser resolution
                pyramid_s = PyramidList_s[c_ + d]
                
                c, h, w = pyramid_c.shape
                interpolated_s = torchvision.transforms.Resize((h, w), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(pyramid_s)
                
                csd.append(torch.abs(pyramid_c - interpolated_s))
                
        return csd

    def normalize_map(self, input_map: torch.Tensor) -> torch.Tensor:
        """Normalizes the input map by the factor :math:`(M-\mu)^2`, 
        where :math:`M` is the maximum of the map and :math:`\mu` the mean of the map.
        
        .. math:: \mathcal{N}(\mathbf{I}(x, y)) = (M-\mu)^2 \cdot \mathbf{I}(x, y)

        Args:
            input_map (torch.Tensor): input map

        Raises:
            TypeError: If input is not of type `torch.Tensor`
            USerWarning: If input map has more than one color channel

        Returns:
            torch.Tensor: normalized map
        """
        # Check if input is of correct type
        if not isinstance(input_map, torch.Tensor):
            raise TypeError(tc.err + "Input must be of type 'torch.Tensor'!")
        # Check if the input is a map with only one color channel
        if len(input_map.shape) > 2 and input_map.shape[0] > 1:
            raise UserWarning(tc.warn + "Input map should only have one color channel, but got more than 1!")
        
        # Normalize Values:
        normalized_map = (input_map - torch.min(input_map)) / (torch.max(input_map) - torch.min(input_map))
        M = torch.max(normalized_map)  # = 1
        mu = torch.mean(normalized_map)
        factor = (M - mu)**2
        return factor * normalized_map 
        
    def linear_combination(self, I_dash: torch.Tensor, C_dash: torch.Tensor, O_dash: torch.Tensor) -> torch.Tensor:
        """Calculates the saliency map S by the following formula:
        
        .. math:: S = \\frac{1}{3}\left(\mathcal{N}(\\bar{I}) + \mathcal{N}(\\bar{C}) + \mathcal{N}(\\bar{O}) \\right)

        Args:
            I_dash (torch.Tensor): across-scale combinations of the intensity maps 
            C_dash (torch.Tensor): across-scale combinations of the color maps
            O_dash (torch.Tensor): across-scale combinations of the orientation maps
            
        Raises:
            TypeError: If one of the inputs in not of type `torch.Tensor`

        Returns:
            torch.Tensor: saliency map S
        """
        # Check if all inputs are of type torch.Tensor
        if isinstance((I_dash, C_dash, O_dash), torch.Tensor):
            raise TypeError(tc.err + "Input arguments must be of type `torch.Tensor`!")
        
        # Normalize input maps 
        N_of_I = self.normalize_map(I_dash) 
        N_of_C = self.normalize_map(C_dash) 
        N_of_O = self.normalize_map(O_dash) 
        
        # Average and return to user
        return (N_of_I + N_of_C + N_of_O) / 3.0
    
    def saliency_map(self, img: torch.Tensor) -> torch.Tensor:
        """Calculates the Saliency map using the Itti's method.
        
        For that purpose the subfunctions of this class will be used, to first calculate the features, then the center-surround differences, 
        which are followed by the across-scale combination and in the end the linear combination of the three maps. 
        
        Everything that this function calculates can be also done manually, if the corresponding subfunctions of this class are called in the right order. 

        Example:
            >>> import torchvision
            >>> img = torchvision.io.read_image("data/cat.png")[0:3, ...] / 255.0
            >>> ef = IttisMethod()
            >>> sm = ef.saliency_map(img)

        Args:
            img (torch.Tensor): input rgb image

        Returns:
            torch.Tensor: saliency map
        """
        # Check, if user inputs right format!
        self.__check_image_dimensions(img)
        c, h, w = img.shape
        
        #########################
        #   linear filtering    #
        #########################
        # Perform linear filtering -> yields features: "colors", "intensity" and "orientations" in pyramid form
        Ip, O, Rp, Gp, Bp, Yp = self.linear_filtering(img)
        
        # Calculate RG and BY for center-surrounded difference
        RG = [Rp[i] - Gp[i] for i in range(len(Rp))]
        GR = [Gp[i] - Rp[i] for i in range(len(Rp))]
        BY = [Bp[i] - Yp[i] for i in range(len(Rp))]
        YB = [Yp[i] - Bp[i] for i in range(len(Rp))]
        
        #####################################
        #   center-surrounded differences   #
        #####################################
        # Calculate center-surrounded differences for the INTENSITY features
        intensity_maps = self.center_surrounded_differences(Ip, Ip)
        
        # Calculate center-surrounded differences for the COLOR features
        RGcs = self.center_surrounded_differences(RG, GR)
        BYcs = self.center_surrounded_differences(BY, YB)
        # Combine both lists
        color_maps = RGcs + BYcs
        
        # Calculate center-surrounded differences for the ORIENTATION features
        # Shape: (theta, csd)
        orientation_maps = []
        for theta_ in range(0, len(self.theta)):
            tmp_csd = self.center_surrounded_differences(O[theta_], O[theta_])
            orientation_maps.append(tmp_csd)

        ##################################################
        #   across-scale combination and normalization   #
        ##################################################
        up_down_scaler = torchvision.transforms.Resize((h // 2**4, w // 2**4))
        
        I_dash = torch.zeros((h // 2**4, w // 2**4))
        C_dash = torch.zeros((h // 2**4, w // 2**4))
        O_dash = torch.zeros((h // 2**4, w // 2**4))
        
        for imap in intensity_maps:
            I_dash += up_down_scaler(self.normalize_map(imap))[0, ...]
        
        for c_idx in range(0, len(RGcs)):
            C_dash += up_down_scaler(self.normalize_map(RGcs[c_idx]))[0, ...] + up_down_scaler(self.normalize_map(BYcs[c_idx]))[0, ...]
        
        # Loop over all orientations
        for ocsd in orientation_maps:
            # Add up all scales for each orientation
            O_dash_theta = torch.zeros((h // 2**4, w // 2**4))
            for omap in ocsd:
                O_dash_theta += up_down_scaler(self.normalize_map(omap))[0, ...]
            
            # Add up over all orientations
            O_dash += self.normalize_map(O_dash_theta)
        
        #############################################
        #   linear combination of all feature maps  #
        #############################################
        sm_intern = self.linear_combination(I_dash=I_dash, C_dash=C_dash, O_dash=O_dash)
                
        # Return!
        return sm_intern
    

if __name__ == '__main__':
    ef = IttisMethod()
    # sm= ef.saliency_map(img)
    print(tc.success + "finished!")