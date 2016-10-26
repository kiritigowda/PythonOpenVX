from goprohero import GoProHero
import io
from PIL import Image
import urllib2


class GoProCamera(GoProHero):
    def __init__(self):
        GoProHero.__init__(self, password='hero3PerC')
        self.npics = self.status().get('npics')
        self.url = 'http://10.5.5.9:8080/videos/DCIM/100GOPRO/'

    def shoot_still(self):
        self.command('mode', 'still')
        #while not (self.status().get('mode') == 'still'):
        #    pass
        self.command('record', 'on')
        #while self.status().get('npics') == self.npics:
        #    pass
        self.npics = self.status().get('npics')

    def get_images_list(self):
        images_list = []
        url = urllib2.urlopen(self.url)
        html = url.read().split()
        for i in html:
            if 'JPG' in i:
                images_list.append(i.split("\"")[1])
        return images_list

    def get_image(self):
        images = self.get_images_list()
        images.sort()
        fd = urllib2.urlopen(self.url + images[-1])
        image = io.BytesIO(fd.read())
        im = Image.open(image)
        return im
