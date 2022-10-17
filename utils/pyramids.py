import torch
import torchvision
from .tcolors import terminalColor as tc


class GaussianPyramids:
    """Generates the gaussian pyramid images for a given image with shape (C, H, W). 
    
    Pyramid images will be generated for each channel separately. 
    """
    def __init__(self, sigma: list=[0, 1, 2, 3, 4, 5, 6, 7, 8]) -> None:
        """Initializes the class. Sigma are the scales, that should be used!

        Args:
            sigma (list, optional): scales. Defaults to [0, 1, 2, 3, 4, 5, 6, 7, 8].
        """
        # Scales (8 octaves 1:1 -> scale 0 up to 1:256 -> scale 8)
        self.sigma: list = sigma
        
        #  Construct kernel for gaussian filtering 
        self.kernel_size = 5

    def generate(self, x: torch.Tensor) -> list:
        """Generates gaussian pyramids images for each channel ``c``.
        
        Output is a list of of list, in which the first (outer dim) iterates over 
        the channels and the following axis over the pyramids images: ``output[channels][pyramids]``
        
        Example:
            >>> pyramids_generator = GaussianPyramids()
            >>> pyramids = pyramids_generator.generate(img) # img == rgb image
            >>> # Get pyramid image of scale 4 from channel 0
            >>> p_img = pyramids[0][4]   # Will be a torch.Tensor

        Args:
            x (torch.Tensor): Input image in shape (C, H, W)

        Raises:
            ValueError: If shape of ``x`` is incorrect
            ValueError: if order of axes are incorrect e.g. Numpy format (H, W, C)

        Returns:
            list: of channels of pyramids -> outer dim: channel, inner dim: pyramids
        """
        if len(x.shape) != 3:
            raise ValueError(tc.err + "input images must be in shape of: (C, H, W)!")
        if x.shape[0] > x.shape[1] or x.shape[0] > x.shape[2]:
             raise ValueError(tc.err + "Expect the format: (C{hannel}, H{eight}, W{idth})!")
        
        c, h, w = x.shape
        
        pyramids = []
        current_pyramid_image = x
        
        # Append first image, that isn't resized!
        transform_network = torch.nn.Sequential(
            torchvision.transforms.GaussianBlur(self.kernel_size)
        )
        pyramids.append(transform_network(x.view((1, c, h, w))).view((c, h, w)))
        
        # Iterate over every scale
        for s in self.sigma[1:]:
            transform_network = torch.nn.Sequential(
                torchvision.transforms.GaussianBlur(self.kernel_size),
                torchvision.transforms.Resize((h // 2**s, w // 2**s))
            )
            current_pyramid_image = transform_network(current_pyramid_image.view((1, c, h // 2**(s - 1), w // 2**(s - 1))))
            pyramids.append(current_pyramid_image.view((c, h // 2**s, w // 2**s)))
        
        return pyramids
     

class GaborPyramids:
    def __init__(self, sigma: list=[0, 1, 2, 3, 4, 5, 6, 7, 8], theta: list = [0, 45, 90, 135]) -> None:
        """Initializes the class. Sigma are the scales, that should be used!

        Args:
            sigma (list, optional): scales. Defaults to [0, 1, 2, 3, 4, 5, 6, 7, 8].
            theta (list, optional): orientations of Gabor filter (degree). Defaults to [0, 45, 90, 135]
        """
        # Scales (8 octaves 1:1 -> scale 0 up to 1:256 -> scale 8)
        self.sigma: list = sigma
        
        self.theta = [t * 3.14159 / 180.0 for t in theta]    # theta [rad] = theta [deg] * pi [rad] / 180 [deg]
        
        self.bank = self.__generate_gabor_filter_bank(2, self.theta, 4.0, 0.0, 1.0)
    
    @staticmethod
    def __generate_gabor_filter_bank(sigma: float, thetas: list, Lambda: float, psi: float, gamma: float) -> list:
        """Copied from https://en.wikipedia.org/wiki/Gabor_filter and changed to torch Framework.

        Args:
            sigma (float): gaussian std
            thetas (list): (list of angles) orientation of the normal to the parallel stripes of the Gabor function
            Lambda (float): wavelength
            psi (float): phase offset
            gamma (float): spatial aspect ratio

        Returns:
            list: Kernels for different orientations
        """
        # Create Blank list
        gb_filter_bank = []
        
        # Iterate over all orientations
        for theta in torch.tensor(thetas):
            sigma_x = sigma
            sigma_y = float(sigma) / gamma

            # Bounding box
            nstds = 3  # Number of standard deviation sigma
            xmax = torch.max(torch.abs(nstds * sigma_x * torch.cos(theta)), torch.abs(nstds * sigma_y * torch.sin(theta)))
            xmax = torch.ceil(max(1, xmax))
            ymax = torch.max(torch.abs(nstds * sigma_x * torch.sin(theta)), torch.abs(nstds * sigma_y * torch.cos(theta)))
            ymax = torch.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = torch.meshgrid(torch.arange(ymin, ymax + 1), torch.arange(xmin, xmax + 1), indexing='ij')

            # Rotation
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(2 * torch.pi / Lambda * x_theta + psi)
            
            gb_filter_bank.append(gb)
        return gb_filter_bank
            
    def __convolve_img(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Convolves the input image ``img`` with given ``kernel``. 
        The fist axis will be untouched and the convolution will be performed for each channel in the first axis separately. 

        Args:
            img (torch.Tensor): input image
            kernel (torch.Tensor): kernel for convolution

        Returns:
            torch.Tensor: convolved image
        """
        c, h, w = img.shape
        
        # Prepare kernel
        k_size = kernel.shape[0]
        ok = kernel.view((1, 1, k_size, k_size))
        
        # Create array for saving convolutions over channels
        result = torch.zeros((c, h, w))
        
        # Loop over each channel and convolve with kernel
        for ch in range(0, c):
            result[ch, ...] = torch.nn.functional.conv2d(img[ch, ...].view((1, 1, h, w)), ok, padding='same')[0, 0, ...]
            
        return result
            
    def generate(self, x: torch.Tensor) -> list:
        """Generates Gabor pyramids for a given image ``x``. 
        The resulting output will be a ``list``, where in the first dimension the orientations and
        in the second dimension the pyramid images are stored. 

        Example:
            >>> pyramids_generator = GaborPyramids()
            >>> pyramids = pyramids_generator.generate(img) # img == rgb image
            >>> # Get pyramid image for orientation 1 and scale 3
            >>> p_img = pyramids[1][3]   # Will be a torch.Tensor of shape [C, H, W]

        Args:
            x (torch.Tensor): Input image in shape (C, H, W)

        Raises:
            ValueError: If shape of ``x`` is incorrect
            ValueError: if order of axes are incorrect e.g. Numpy format (H, W, C)

        Returns:
            list: pyramids (list) in form of ``list[orientations][pyramids] -> torch.Tensor``
        """
        if len(x.shape) != 3:
            raise ValueError(tc.err + "input images must be in shape of: (C, H, W)!")
        if x.shape[0] > x.shape[1] or x.shape[0] > x.shape[2]:
             raise ValueError(tc.err + "Expect the format: (C{hannel}, H{eight}, W{idth})!")
         
        orientation_pyramids = []
        c, h, w = x.shape
        
        for kernel in self.bank:
            # Create empty array for pyramids
            pyramids = []
            
            # Start with the original image and then create Pyramid images
            current_image = x
            for s in self.sigma:
                convolved_img = self.__convolve_img(current_image, kernel)                
                current_image = torchvision.transforms.Resize((h // 2**s, w // 2**s))(convolved_img)
                pyramids.append(current_image)
            
            # Save the pyramid images for the current orientation
            orientation_pyramids.append(pyramids)

        return orientation_pyramids


if __name__ == '__main__':
    print(tc.info + "Please run a different file!")