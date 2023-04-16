class Config:

    def __init__(self, **kwargs):

        self.feature_pixels = 16

        self.anchors_areas = [4*4, 8*8, 16*16, 32*32, 64*64, 128*128, 256*256]
        self.aspect_ratios = [1 / 2, 1 / 1, 2 / 1]
        self.num_anchors = len(self.anchors_areas) * len(self.aspect_ratios)

        self.max_image_size = 800
        self.augment = True

        self.use_horizontal_flips = True
        self.use_vertical_flips = True
        self.use_rot_90 = True

        self.__dict__.update(kwargs)