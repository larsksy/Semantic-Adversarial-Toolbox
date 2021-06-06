from PIL import Image
from sat.defence.Defence import ProcessingDefence
import torchvision.transforms as transform
from io import BytesIO


class JPEGCompression(ProcessingDefence):
    """Implementation of JPEG Compression defence"""

    def __init__(self, quality=20):
        """

        :param quality: Compression quality. Lower means more compression.
        """
        super(JPEGCompression).__init__()

        assert 100 >= quality > 0

        self.quality = quality
        self.to_image = transform.ToPILImage()
        self.to_tensor = transform.ToTensor()

    def __call__(self, image):
        """

        :param image: The image to apply the defence to.
        :return: Image tensor compressed using JPEG compression.
        """

        if image.ndim == 4:
            image = image.squeeze()

        jpg = BytesIO()
        pil = self.to_image(image)
        pil.save(jpg, 'JPEG', quality=self.quality)
        tensor = Image.open(jpg)
        tensor = self.to_tensor(tensor)

        return tensor
