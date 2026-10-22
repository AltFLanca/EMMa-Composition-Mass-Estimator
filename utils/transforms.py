import albumentations as A

class CustomHorizontalFlip(A.HorizontalFlip):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.applied = False

    def apply(self, img, **params):
        self.applied = True
        return super().apply(img, **params)

    def apply_to_mask(self, img, **params):
        self.applied = True
        return super().apply_to_mask(img, **params)

    def clear(self):
        self.applied = False


class CustomVerticalFlip(A.VerticalFlip):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.applied = False

    def apply(self, img, **params):
        self.applied = True
        return super().apply(img, **params)

    def apply_to_mask(self, img, **params):
        self.applied = True
        return super().apply_to_mask(img, **params)

    def clear(self):
        self.applied = False