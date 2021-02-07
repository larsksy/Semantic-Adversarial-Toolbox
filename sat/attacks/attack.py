
class Attack:
    pass


class ColorAttack(Attack):
    pass



class Adversarial():

    def __init__(self, image, image_adv, label, label_adv):
        self.image = image
        self.image_adv = image_adv
        self.label = label
        self.label_adv = label_adv




class AdversarialSet():

    def __init__(self):
        self.set = set()

    def add_adversarial(self, adv):
        self.set.add(adv)
