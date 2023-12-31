import platform
import matplotlib
if platform.system() == 'Linux':
    matplotlib.use('Agg')
import numpy
from numpy import fft
from scipy.optimize import fmin_cg
from scipy import ndimage, signal
import copy
import matplotlib.pyplot as plt
import tool_box
from matplotlib import cm

class Fourier_Quad:

    def __init__(self, size, seed):
        self.rng = numpy.random.RandomState(seed)
        self.size = size
        self.alpha = (2.*numpy.pi/size)**4

        self.ky = numpy.mgrid[0: size, 0: size][0] - size/2.
        self.kx = numpy.mgrid[0: size, 0: size][1] - size/2.

        self.kx2 = self.kx*self.kx
        self.ky2 = self.ky*self.ky
        self.kxy = self.kx*self.ky
        self.k2 = self.kx2+self.ky2
        self.k4 = self.k2*self.k2
        self.mn1 = (-0.5)*(self.kx2 - self.ky2)
        self.mn2 = -self.kx*self.ky
        self.mn4 = self.k4 - 8*self.kx2*self.ky2
        self.mn5 = self.kxy*(self.kx2 - self.ky2)

        self.my = self.ky - 0.5
        self.mx = self.kx - 0.5

        self.rim = self.border(1)
        self.flux2 = -1.
        self.hlr = 0


    def draw_noise(self, mean, sigma):
        noise_img = self.rng.normal(loc=mean, scale=sigma, size=self.size * self.size).reshape(self.size, self.size)
        return noise_img

    def pow_spec(self, image):
        image_ps = fft.fftshift((numpy.abs(fft.fft2(image)))**2)
        return image_ps

    def pow_real_imag(self,image):
        img_f = fft.fftshift(fft.fft2(image))
        return img_f.real, img_f.imag

    def pow_arg(self, image):
        return numpy.angle(fft.fftshift(fft.fft2(image)))

    def shear_est(self, gal_image, psf_image, noise=None, F=False):
        r"""
        Before the shear_est(), the get_hlr() should be called to get the half light radius of the PSF, and it will
        assigned to the self.hlr for shear_est() automatically. The get_hlr() should be called when the PSF changes.
        :param gal_image: galaxy image
        :param psf_image: PSF image or Power Spectrum of PSF image
        :param noise: noise image
        :param F: boolean, False for PSF image, True for PSF Power spectrum
        :return: estimators of shear
        """
        # gal_ps = self.pow_spec(gal_image)
        gal_ps = gal_image
        # gal_ps = tool_box.smooth(gal_ps,self.size)
        if noise is not None:
            nbg = self.pow_spec(noise)
            self.flux2 = numpy.sqrt(gal_ps[int(self.size/2), int(self.size/2)]/numpy.sum(self.rim*gal_ps)*numpy.sum(self.rim))
            # nbg = tool_box.smooth(nbg,self.size)
            # rim = self.border(2, size)
            # n = numpy.sum(rim)
            # gal_pn = numpy.sum(gal_ps*rim)/n                # the Possion noise of galaxy image
            # nbg_pn = numpy.sum(nbg*rim)/n                   # the  Possion noise of background noise image
            gal_ps = gal_ps - nbg# + nbg_pn - gal_pn

        if F:
            psf_ps = psf_image
        else:
            psf_ps = self.pow_spec(psf_image)
        #self.get_radius_new(psf_ps, 2)
        wb, beta = self.wbeta(self.hlr)
        maxi = numpy.max(psf_ps)
        idx = psf_ps < maxi / 100000.
        wb[idx] = 0
        psf_ps[idx] = 1.
        tk = wb/psf_ps * gal_ps

        # ky, kx = self.ky, self.kx
        # #
        # kx2 = kx*kx
        # ky2 = ky*ky
        # kxy = kx*ky
        # k2 = kx2 + ky2
        # k4 = k2*k2
        # mn1 = (-0.5)*(kx2 - ky2)    # (-0.5)*(kx**2 - ky**2)
        # mn2 = -kxy                  # -kx*ky
        # mn3 = k2 - 0.5*beta**2*k4   # kx**2 + ky**2 - 0.5*beta**2*(kx**2 + ky**2)**2
        # mn4 = k4 - 8*kx2*ky2        # kx**4 - 6*kx**2*ky**2 + ky**4
        # mn5 = kxy*(kx2 - ky2)       # kx**3*ky - kx*ky**3

        # mn1 = self.mn1
        # mn2 = self.mn2
        mn3 = self.k2 - 0.5*beta**2*self.k4
        # mn4 = self.mn4
        # mn5 = self.mn5

        mg1 = numpy.sum(self.mn1 * tk)*self.alpha
        mg2 = numpy.sum(self.mn2 * tk)*self.alpha
        mn  = numpy.sum(mn3 * tk)*self.alpha
        mu  = numpy.sum(self.mn4 * tk)*(-0.5*beta**2)*self.alpha
        mv  = numpy.sum(self.mn5 * tk)*(-2.*beta**2)*self.alpha

        return mg1, mg2, mn, mu, mv

    def wbeta(self, radius):
        w_temp = numpy.exp(-(self.kx**2 + self.ky**2)/radius**2)
        return w_temp, 1./radius

    def ran_pts(self, num, radius, step=1, ellip=0, alpha=0, g=None):
        """
        create random points within a ellipse
        :param num: point number
        :param radius: max radius for a point from the center
        :param ellip: ellipticity
        :param alpha: radian, the angle between the major-axis and x-axis
        :param g: shear, tuple or list or something can indexed by g[0] and g[1]
        :return: (2,num) numpy array, the points
        """
        r2 = radius ** 2
        b2 = (1 - ellip) / (1 + ellip)
        cov1, cov2 = numpy.cos(alpha), numpy.sin(alpha)
        xy_coord = numpy.zeros((2, num))
        theta = self.rng.uniform(0., 2 * numpy.pi, num)
        xn = numpy.cos(theta)*step
        yn = numpy.sin(theta)*step
        x = 0.
        y = 0.
        for n in range(num):
            x += xn[n]
            y += yn[n]
            if x ** 2 + y ** 2 / b2 > r2:
                x = xn[n]
                y = yn[n]
            xy_coord[0, n] = x * cov1 - y * cov2
            xy_coord[1, n] = x * cov2 + y * cov1

        xy_coord[0] = xy_coord[0] - numpy.mean(xy_coord[0])
        xy_coord[1] = xy_coord[1] - numpy.mean(xy_coord[1])

        if g:
            sheared = numpy.dot(numpy.array(([(1 + g[0]), g[1]], [g[1], (1 - g[0])])), xy_coord)
            return xy_coord, sheared
        else:
            return xy_coord

    def ran_pts_e(self, num, radius, step=2.5, xy_ratio=1.):
        """
        create random points within a ellipse
        :param num: point number
        :param radius: max radius for a point from the center
        :param ellip: ellipticity
        :param alpha: radian, the angle between the major-axis and x-axis
        :param g: shear, tuple or list or something can indexed by g[0] and g[1]
        :return: (2,num) numpy array, the points
        """
        r2 = radius ** 2
        xy_coord = numpy.zeros((2, num))
        theta = self.rng.uniform(0., 2 * numpy.pi, num)
        if xy_ratio >= 1:
            xn = numpy.cos(theta) * step * xy_ratio
            yn = numpy.sin(theta) * step
        else:
            xn = numpy.cos(theta) * step
            yn = numpy.sin(theta) * step / xy_ratio
        x = 0.
        y = 0.
        for n in range(num):
            x += xn[n]
            y += yn[n]
            if x ** 2 + y ** 2 > r2:
                x = xn[n]
                y = yn[n]
            xy_coord[0, n] = x
            xy_coord[1, n] = y

        xy_coord[0] = xy_coord[0] - numpy.mean(xy_coord[0])
        xy_coord[1] = xy_coord[1] - numpy.mean(xy_coord[1])

        return xy_coord

    def rotate(self, pos, theta):
        rot_matrix = numpy.array([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]])
        return numpy.dot(rot_matrix, pos)

    # def shear(self, pos, g1, g2):
    #     return numpy.dot(numpy.array(([1+g1, g2], [g2, 1-g1])), pos)
    def shear(self, pos, g1, g2):
        return numpy.dot(numpy.array(([(1+g1), g2], [g2, (1-g1)])), pos)
    def rotation(self, e1, e2, theta):
        """
        return the new (e1,e2) after rotation
        e = (e1 + i e2)*exp(2i\theta)
        :param e1: ellipticity
        :param e2: ellipticity
        :param theta: position angle
        :return:
        """
        e1_r = e1 * numpy.cos(2 * theta) - e2 * numpy.sin(2 * theta)
        e2_r = e1 * numpy.sin(2 * theta) + e2 * numpy.cos(2 * theta)
        return e1_r, e2_r

    def convolve_psf(self, pos, psf_scale, flux=1.,img_cent=(0,0), ellip_theta=None, psf="Moffat"):
        # type: #(object, object, object, object, object, object) -> object
        # type: #(object, object, object, object, object, object) -> object
        pst_num = pos.shape[1]
        arr = numpy.zeros((self.size, self.size))
        dx, dy = img_cent
        if psf == "GAUSS":
            factor = flux/2/numpy.pi/psf_scale**2
            for i in range(pst_num):
                arr += factor*numpy.exp(-((self.mx-pos[0, i] - dx)**2+(self.my-pos[1, i] -dy)**2)/2./psf_scale**2)

        elif psf == "Moffat":

            # factor = flux/pst_num
            for i in range(pst_num):
                x,y = pos[0,i],pos[1,i]
                pts_psf = self.cre_psf(psf_scale, flux,img_cent=(x+dx,y+dy), ellip_theta=ellip_theta, model=psf)
                arr += pts_psf
        return arr

    def cre_psf(self, psf_scale, flux=1.,img_cent=(0,0), ellip_theta=None, model="Moffat"):
        # dx,dy is shift from the original image cernt [image_size/2-0.5,image_size/2-0.5] (not [0,0] !)
        dx,dy = img_cent
        if model == 'GAUSS':
            factor = flux*1./2/numpy.pi/psf_scale**2
            arr = factor*numpy.exp(-((self.mx-dx)**2 + (self.my-dy)**2)/2./psf_scale**2)
            return arr

        elif model == 'Moffat':
            r_scale_sq = 9
            m = 3.5
            if ellip_theta:
                ellip, theta = ellip_theta
                cos_theta = numpy.cos(theta)
                sin_theta = -numpy.sin(theta)
                # [cos \theta, -sin \theta]
                # [sin \theta, cos \theta]
                q2 = (1 - ellip) / (1 + ellip)

                # x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1, (a > b)
                # q = b / a, q ^ 2 = (1 - e) / (1 + e)
                # = > q ^ 2
                # x^ 2 / b ^ 2 + y ^ 2 / b ^ 2 = 1

                b2_inv = 1. / psf_scale / psf_scale
                a2_inv = b2_inv * q2
                factor = flux*numpy.sqrt(q2)/(numpy.pi*psf_scale**2*(1 - (1. + r_scale_sq)**(1.-m))/(m-1.))

                rsq = (cos_theta*(self.mx-dx) - sin_theta*(self.my-dy))**2*a2_inv + \
                      (sin_theta*(self.mx-dx) + cos_theta*(self.my-dy))**2*b2_inv
                idx = rsq > r_scale_sq
                rsq[idx] = 0.
                arr = factor*(1. + rsq)**(-m)
                arr[idx] = 0.

            else:
                factor = flux*1./(numpy.pi*psf_scale**2*(1 - (1. + r_scale_sq)**(1.-m))/(m-1.))
                rsq = ((self.mx-dx)**2 + (self.my-dy)**2) / psf_scale**2
                idx = rsq > r_scale_sq
                rsq[idx] = 0.
                arr = factor*(1. + rsq)**(-m)
                arr[idx] = 0.
            return arr

    def get_radius(self, image, scale):
        # get the radius of the flux descends to the maximum/scale
        radi_arr = copy.copy(image)
        maxi = numpy.max(radi_arr)
        y, x = numpy.where(radi_arr == maxi)
        idx = radi_arr < maxi / scale
        radi_arr[idx] = 0.
        idx = radi_arr > 0.
        radi_arr[idx] = 1.
        half_radi_pool = []
        half_radi_pool.append((int(y[0]), int(x[0])))

        def check(x):  # x is a two components  tuple -like input
            if (x[0] - 1, x[1]) not in half_radi_pool and radi_arr[x[0] - 1, x[1]] == 1:
                half_radi_pool.append((x[0] - 1, x[1]))
            if (x[0] + 1, x[1]) not in half_radi_pool and radi_arr[x[0] + 1, x[1]] == 1:
                half_radi_pool.append((x[0] + 1, x[1]))
            if (x[0], x[1] + 1) not in half_radi_pool and radi_arr[x[0], x[1] + 1] == 1:
                half_radi_pool.append((x[0], x[1] + 1))
            if (x[0], x[1] - 1) not in half_radi_pool and radi_arr[x[0], x[1] - 1] == 1:
                half_radi_pool.append((x[0], x[1] - 1))
            return len(half_radi_pool)

        while True:
            for cor in half_radi_pool:
                num0 = len(half_radi_pool)
                num1 = check(cor)
            if num0 == num1:
                break
        self.hlr = numpy.sqrt(len(half_radi_pool) / numpy.pi)
        return numpy.sqrt(len(half_radi_pool) / numpy.pi)

    def get_radius_new(self, image, scale):
        # get the radius of the flux descends to the maximum/scale
        radi_arr = copy.copy(image)
        maxi = numpy.max(radi_arr)
        y, x = numpy.where(radi_arr == maxi)
        idx = radi_arr < maxi / scale
        radi_arr[idx] = 0.
        half_radius_pool = []
        flux = []

        def detect(mask, ini_y, ini_x, signal, signal_val, size):
            if mask[ini_y, ini_x] > 0:
                signal.append((ini_y, ini_x))
                signal_val.append(mask[ini_y, ini_x])
                mask[ini_y, ini_x] = 0
                for cor in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if -1 < ini_y + cor[0] < size and -1 < ini_x + cor[1] < size \
                            and mask[ini_y + cor[0], ini_x + cor[1]] > 0:
                        detect(mask, ini_y + cor[0], ini_x + cor[1], signal, signal_val, size)
            return signal, signal_val

        half_radius_pool, flux = detect(radi_arr, y[0], x[0], half_radius_pool, flux, self.size)
        self.hlr = numpy.sqrt(len(half_radius_pool)/numpy.pi)
        return numpy.sqrt(len(half_radius_pool)/numpy.pi), half_radius_pool, numpy.sum(flux), maxi, (y, x)

    def move(self, image, x, y):
        imagesize = image.shape[0]
        cent = numpy.where(image == numpy.max(image))
        dx = int(numpy.max(cent[1]) - x)
        dy = int(numpy.max(cent[0]) - y)
        if dy > 0:
            if dx > 0:
                arr_y = image[0:dy, 0:imagesize]
                arr = image[dy:imagesize, 0:imagesize]
                arr = numpy.row_stack((arr, arr_y))
                arr_x = arr[0:imagesize, 0:dx]
                arr = arr[0:imagesize, dx:imagesize]
                arr = numpy.column_stack((arr, arr_x))
                return arr
            elif dx == 0:
                arr_y = image[0:dy, 0:imagesize]
                arr = image[dy:imagesize, 0:imagesize]
                arr = numpy.row_stack((arr, arr_y))
                return arr
            else:
                arr_y = image[0:dy, 0:imagesize]
                arr = image[dy:imagesize, 0:imagesize]
                arr = numpy.row_stack((arr, arr_y))
                arr_x = arr[0:imagesize, 0:imagesize + dx]
                arr = arr[0:imagesize, imagesize + dx:imagesize]
                arr = numpy.column_stack((arr, arr_x))
                return arr
        elif dy == 0:
            if dx > 0:
                arr_x = image[0:imagesize, 0:dx]
                arr = image[0:imagesize, dx:imagesize]
                arr = numpy.column_stack((arr, arr_x))
                return arr
            elif dx == 0:
                return image
            else:
                arr = image[0:imagesize, 0:imagesize + dx]
                arr_x = image[0:imagesize, imagesize + dx:imagesize]
                arr = numpy.column_stack((arr_x, arr))
                return arr
        elif dy < 0:
            if dx > 0:
                arr_y = image[imagesize + dy:imagesize, 0:imagesize]
                arr = image[0:imagesize + dy, 0:imagesize]
                arr = numpy.row_stack((arr_y, arr))
                arr_x = arr[0:imagesize, 0:dx]
                arr = arr[0:imagesize, dx:imagesize]
                arr = numpy.column_stack((arr, arr_x))
                return arr
            elif dx == 0:
                arr_y = image[imagesize + dy:imagesize, 0:imagesize]
                arr = image[0:imagesize + dy, 0:imagesize]
                arr = numpy.row_stack((arr_y, arr))
                return arr
            else:
                arr_y = image[imagesize + dy:imagesize, 0:imagesize]
                arr = image[0:imagesize + dy, 0:imagesize]
                arr = numpy.row_stack((arr_y, arr))
                arr_x = arr[0:imagesize, 0:imagesize + dx]
                arr = arr[0:imagesize, imagesize + dx:imagesize]
                arr = numpy.column_stack((arr, arr_x))
                return arr

    def get_centroid(self, image, filt=False, radius=2):
        y0,x0 = self.mfpoly(image)
        y, x = numpy.mgrid[0:image.shape[0], 0:image.shape[1]]
        # m0 = numpy.sum(image)
        # mx  = numpy.sum(x * image)
        # my  = numpy.sum(y * image)
        # x0 = mx / m0
        # y0 = my / m0
        #yc,xc = numpy.where(image==numpy.max(image))
        y = y - y0
        x = x - x0
        if filt == True:
            gaus_filt = numpy.exp(-(y**2+x**2)/2/numpy.pi/radius**2)
            image = image *gaus_filt
        mxx = numpy.sum(x*x*image)
        mxy = numpy.sum(x * y * image)
        myy = numpy.sum(y* y * image)
        e1 = (mxx-myy)/(mxx+myy)
        e2 = 2*mxy/(mxx+myy)
        return y0, x0,e1,e2

    def psf_align(self, image):
        imagesize = image.shape[0]
        arr = self.move(image, 0, 0)
        image_f = fft.fft2(arr)
        yy, xx = numpy.mgrid[0:48, 0:48]
        xx = numpy.mod(xx + 24, 48) - 24
        yy = numpy.mod(yy + 24, 48) - 24
        fk = numpy.abs(image_f) ** 2
        line = numpy.sort(numpy.array(fk.flat))
        idx = fk < 4 * line[int(imagesize ** 2/ 2)]
        fk[idx] = 0
        weight = fk / line[int(imagesize ** 2/ 2)]
        kx = xx * 2 * numpy.pi / imagesize
        ky = yy * 2 * numpy.pi / imagesize

        def pha(p):
            x, y = p
            return numpy.sum((numpy.angle(image_f*numpy.exp(-1.0j*(kx*x+ky*y))))**2*weight)

        res = fmin_cg(pha, [0, 0], disp=False)
        inve = fft.fftshift(numpy.real(fft.ifft2(image_f*numpy.exp(-1.0j*(kx*res[0]+ky*res[1])))))
        return inve

    def Qfilter(self, psfimage):
        x, y = numpy.mgrid[0:27, 0:27]
        xc, yc = 13.1, 13.1
        # w = 1.3
        # ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker = 1/(3.14*((x-xc)**2+(y-yc)**2))*(1-(1+((x-xc)**2+(y-yc)**2)/(1.5)**2)*numpy.exp(-((x-xc)**2+(y-yc)**2)/(1.5)**2))\
              *numpy.exp(-((x-xc)**2+(y-yc)**2)**2/13**4)

        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def sumfilter(self, psfimage,intrange):
        x, y = numpy.mgrid[0:27, 0:27]
        xc, yc = 13.1, 13.1
        dist = numpy.sqrt((x-xc)**2+(y-yc)**2)
        tmp = numpy.ones([27,27])
        tmp[dist>intrange]=0
        sumresult = numpy.sum(psfimage*tmp)
        return sumresult
    def ker2_tmp(self,ker2,d0,d,xc=13.001,yc=13.001):
        # x, y = numpy.mgrid[0:2600, 0:2600]
        # xc, yc = 13.001, 13.001
        # x = x / 100
        # y = y / 100
        # d = numpy.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        tmp = numpy.ones([260, 260])
        tmp[d< d0] = 0

        return numpy.sum(ker2*tmp*0.1*0.1)

    def Ufilter(self,psfimage):
        x, y = numpy.mgrid[0:260, 0:260]
        xc, yc = 12.95, 12.95
        x = x / 10
        y = y / 10
        d = numpy.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        tmp1 = numpy.ones([260, 260])
        tmp1[d <= 0.1] = 0
        tmp2 = numpy.ones([260, 260])
        tmp2[d > 0.1] = 0
        ker2_tmp1 = 2 / (3.14 * (d ** 3)) * (1 - (1 + (d ** 2) / (1.5) ** 2) * numpy.exp(-(d ** 2) / (1.5) ** 2)) \
                    * numpy.exp(-(d ** 2) ** 2 / 13 ** 4)
        ker2_tmp2 = 2 / (3.14) * (0.5*d/1.5**4-d**3/1.5**6)*(1-d**4/13**4)
        ker12 = ker2_tmp1 * tmp1 + ker2_tmp2 * tmp2

        x, y = numpy.mgrid[0:27, 0:27]
        xc, yc = 13.1, 13.1
        ker1 = -1/(3.14*((x-xc)**2+(y-yc)**2))*(1-(1+((x-xc)**2+(y-yc)**2)/(1.5)**2)*numpy.exp(-((x-xc)**2+(y-yc)**2)/(1.5)**2))\
              *numpy.exp(-((x-xc)**2+(y-yc)**2)**2/13**4)

        x, y = numpy.mgrid[0:27, 0:27]
        xc, yc = 13.01, 13.01
        ker2_dist = numpy.sqrt((x-xc)**2+(y-yc)**2)
        ker2 = numpy.zeros([27,27])
        for i in range(27):
            for j in range(27):
                ker2[j,i]=self.ker2_tmp(ker12,ker2_dist[j,i],d)
        ker = ker1+ker2

        # ker = ker/numpy.max(ker)

        U0 = numpy.sum(ker2_dist*ker)/27**2
        ker = ker-U0

        fig, (ax1) = plt.subplots()
        im1 = ax1.imshow(ker, cmap=cm.coolwarm)
        cbar1 = fig.colorbar(im1, shrink=0.5)
        fig.tight_layout()
        fig.savefig('Ukernal_imshow.pdf' )

        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def Ufilter_frombook(self,psfimage):
        x, y = numpy.mgrid[0:27, 0:27]
        xc, yc = 13.001, 13.001
        ker = 1/(9)*(1-((x-xc)**2+(y-yc)**2)/18)*numpy.exp(-((x-xc)**2+(y-yc)**2)/18)
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov

    def gaussfilter(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1.0, 1.0
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov

    def gaussfilter2(self, psfimage):
        x, y = numpy.mgrid[0:5, 0:5]
        xc, yc = 2., 2.
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov

    def gaussfilter3(self, psfimage):
        x, y = numpy.mgrid[0:9, 0:9]
        xc, yc = 4., 4.
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def gaussfilter4(self, psfimage):
        x, y = numpy.mgrid[0:13, 0:13]
        xc, yc = 6., 6.
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def gaussfilter5(self, psfimage):
        x, y = numpy.mgrid[0:15, 0:15]
        xc, yc = 7., 7.
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def gaussfilter6(self, psfimage):
        x, y = numpy.mgrid[0:19, 0:19]
        xc, yc = 9., 9.
        w = 2.0
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov

    def tophatfilter(self, psfimage):
        x, y = numpy.mgrid[0:5, 0:5]
        xc, yc = 2, 2
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:]=1/5/5
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def tophatfilter2(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:]=1/3/3
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def tophatfilter3(self, psfimage):
        x, y = numpy.mgrid[0:25, 0:25]
        xc, yc = 12, 12
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:]=1/25/25
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def tophatfilter4(self, psfimage):
        x, y = numpy.mgrid[0:17, 0:17]
        xc, yc = 8,8
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:]=1/17/17
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def tophatfilter5(self, psfimage):
        x, y = numpy.mgrid[0:15, 0:15]
        xc, yc = 7, 7
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:]=1/15/15
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def tophatfilter6(self, psfimage):
        x, y = numpy.mgrid[0:9, 0:9]
        xc, yc = 4, 4
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:]=1/9/9
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov

    def subfilter1(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:] =0
        ker[1,1]=1
        ker[0,1] = -1
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def subfilter2(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:] =0
        ker[1,1]=1
        ker[1,0] = -1
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def subfilter3(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:] =0
        ker[1,1]=1
        ker[2,1] = -1
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def subfilter4(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:] =0
        ker[1,1]=1
        ker[1,2] = -1
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def subfilter5(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:] =0
        ker[1,1]=1
        ker[0,0] = -1
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def subfilter6(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:] =0
        ker[1,1]=1
        ker[0,2] = -1
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def subfilter7(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:] =0
        ker[1,1]=1
        ker[2,0] = -1
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def subfilter8(self, psfimage):
        x, y = numpy.mgrid[0:3, 0:3]
        xc, yc = 1, 1
        w = 1.3
        ker = (1.0/2.0/numpy.pi/w/w)*numpy.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        ker[:,:] =0
        ker[1,1]=1
        ker[2,2] = -1
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov
    def mfpoly(self,psf):
        p = numpy.where(psf == numpy.max(psf))
        yr, xr = p[0][0], p[1][0]
        yp, xp = numpy.mgrid[yr - 1:yr + 2, xr - 1:xr + 2]
        patch = psf[yr - 1:yr + 2, xr - 1:xr + 2]
        zz = patch.reshape(9)
        xx = xp.reshape(9)
        yy = yp.reshape(9)
        xy = xx * yy
        x2 = xx * xx
        y2 = yy * yy
        A = numpy.array([numpy.ones_like(zz), xx, yy, x2, xy, y2]).T
        cov = numpy.linalg.inv(numpy.dot(A.T, A))
        a, b, c, d, e, f = numpy.dot(cov, numpy.dot(A.T, zz))
        coeffs = numpy.array([[2.0 * d, e], [e, 2.0 * f]])
        mult = numpy.array([-b, -c])
        xc, yc = numpy.dot(numpy.linalg.inv(coeffs), mult)
        return yc, xc

    def segment(self, image):
        shape = image.shape
        y = int(shape[0] / self.size)
        x = int(shape[1] / self.size)
        star = [image[iy * self.size:(iy + 1) * self.size, ix * self.size:(ix + 1) * self.size]
                for iy in range(y) for ix in range(x)]
        for i in range(x):
            if numpy.sum(star[-1]) == 0:
                star.pop()
        return star  # a list of psfs

    def stack(self, image_array, columns):
        # the inverse operation of divide_stamps
        # the image_array is a three dimensional array of which the length equals the number of the stamps
        num = len(image_array)
        row_num, c = divmod(num, columns)
        if c != 0:
            row_num += 1
        arr = numpy.zeros((row_num*self.size, columns * self.size))
        for j in range(row_num):
            for i in range(columns):
                tag = i + j * columns
                if tag > num - 1:
                    break
                arr[j*self.size:(j+1)*self.size, i*self.size:(i+1)*self.size] = image_array[tag]
        return arr

    def stack_new(self, image_array, stamp, iy, ix):
        image_array[iy * self.size:(iy + 1) * self.size, ix * self.size:(ix + 1) * self.size] = stamp

    def fit(self, star_stamp, noise_stamp, star_data, mode=1):
        psf_pool = self.segment(star_stamp)
        noise_pool = self.segment(noise_stamp)
        x = star_data[:, 0]
        y = star_data[:, 1]
        sxx = numpy.sum(x * x)
        syy = numpy.sum(y * y)
        sxy = numpy.sum(x * y)
        sx = numpy.sum(x)
        sy = numpy.sum(y)
        sz = 0.
        szx = 0.
        szy = 0.
        d=1
        rim = self.border(d)
        n = numpy.sum(rim)
        for i in range(len(psf_pool)):
            if mode==1:
                pmax = numpy.max(psf_pool[i])
                arr = psf_pool[i]/pmax#[p[0][0]-1:p[0][0]+1,p[1][0]-1:p[1][0]+1])
                conv = self.gaussfilter(arr)
                dy,dx = self.mfpoly(conv)
                psf = ndimage.shift(arr,(24-dy,24-dx),mode='reflect')
            elif mode==2:
                psf = self.pow_spec(psf_pool[i])
                noise = self.pow_spec(noise_pool[i])
                # psf_pnoise = numpy.sum(rim*psf)/n
                # noise_pnoise = numpy.sum(rim*noise)/n
                psf =psf - noise# -psf_pnoise+noise_pnoise
                pmax = (psf[23,24]+psf[24,23]+psf[24,25]+psf[25,24])/4
                #pmax = numpy.sum(psf)
                psf = psf / pmax
                psf[24,24]=1
            else:
                psf = self.psf_align(psf_pool[i])
                psf = psf/numpy.max(psf)
            sz += psf
            szx += psf * x[i]
            szy += psf * y[i]
        a = numpy.zeros((self.size, self.size))
        b = numpy.zeros((self.size, self.size))
        c = numpy.zeros((self.size, self.size))
        co_matr = numpy.array([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, len(x)]])
        for m in range(self.size):
            for n in range(self.size):
                re = numpy.linalg.solve(co_matr, numpy.array([szx[m, n], szy[m, n], sz[m, n]]))
                a[m, n] = re[0]
                b[m, n] = re[1]
                c[m, n] = re[2]
        return a, b, c

    def border(self, edge):
        if edge >= self.size/2.:
            print("Edge must be smaller than half of  the size!")
        else:
            arr = numpy.ones((self.size, self.size))
            arr[edge: self.size - edge, edge: self.size - edge] = 0.
            return arr

    def snr_f(self, image):
        """
        estimate the SNR in Fourier space
        :param image:
        :return:
        """
        image_ps = self.pow_spec(image)
        noise_level = numpy.sum(self.rim*image_ps)/numpy.sum(self.rim)
        return numpy.sqrt(image_ps[int(self.size/2), int(self.size/2)]/noise_level)

    def set_bin(self, data, bin_num, scale=100., sort_method="abs", sym=True):
        """
        set up bins for 1-D data
        :param data:
        :param bin_num: total number of bins
        :param sort_method: "abs": set the scale of bins according to the absolute value of data
                            "posi": set the scale of bins according to the positive value
                            else: just set scale of bins from the small end to the large end
        :param sym: True: set up bins symmetrically, False: the bins for the positive values
        :return: bins, (N, ) numpy array
        """
        if sort_method == "abs":
            temp_data = numpy.sort(numpy.abs(data))
        elif sort_method == "posi":
            temp_data = numpy.sort(data[data>0])
        else:
            temp_data = numpy.sort(data)

        if sym:
            bin_size = len(temp_data) / bin_num * 2
            bins = numpy.array([temp_data[int(i * bin_size)] for i in range(1, int(bin_num / 2))])
            bins = numpy.sort(numpy.append(numpy.append(-bins, [0.]), bins))
            bound = numpy.max(numpy.abs(data)) * scale
            bins = numpy.append(-bound, numpy.append(bins, bound))
        else:
            bin_size = len(temp_data) / bin_num
            bins = numpy.array([temp_data[int(i * bin_size)] for i in range(1, bin_num)])
            if sort_method == "abs" or sort_method == "posi":
                bins = numpy.sort(numpy.append([0], bins))
                bound = numpy.max(numpy.abs(data)) * scale
                bins = numpy.append(bins, bound)
            else:
                # for the sort of negative data
                bound = numpy.min(data)*scale
                bins = numpy.append(bound, numpy.append(bins, -bound))
        return bins

    def get_chisq(self, g, nu, g_h, bins, bin_num2, inverse, ig_num):  # checked 2017-7-9!!!
        r"""
        to calculate the symmetry the shear estimators
        :param g: estimators from Fourier quad, 1-D numpy array
        :param nu: N + U for g1, N - U for g2, 1-D numpy array
        :param g_h: pseudo shear (guess)
        :param bins: bin of g for calculation of the symmetry, 1-D numpy array
        :param ig_num: the number of inner grid of bin to be neglected
        :return: chi square
        """
        G_h = g - nu * g_h
        num = numpy.histogram(G_h, bins)[0]

        n1 = num[0:bin_num2][inverse]
        n2 = num[bin_num2:]
        xi = (n1 - n2) ** 2 / (n1 + n2)
        return numpy.sum(xi[:len(xi)-ig_num]) * 0.5

    def get_chisq_withweight(self, g, nu, weight, g_h, bins, bin_num2, inverse, ig_num):  # checked 2017-7-9!!!
        r"""
		to calculate the symmetry the shear estimators
		:param g: estimators from Fourier quad, 1-D numpy array
		:param nu: N + U for g1, N - U for g2, 1-D numpy array
		:param g_h: pseudo shear (guess)
		:param bins: bin of g for calculation of the symmetry, 1-D numpy array
		:param ig_num: the number of inner grid of bin to be neglected
		:return: chi square
		"""
        G_h = g - nu * g_h
        num = numpy.histogram(G_h, bins=bins, weights = weight**2)[0]
        n1 = num[0:bin_num2][inverse]
        n2 = num[bin_num2:]

        num2 = numpy.histogram(G_h, bins=bins, weights = weight)[0]
        n12 = num2[0:bin_num2][inverse]
        n22 = num2[bin_num2:]

        xi = (n12 - n22) ** 2 / (n1 + n2)
        return numpy.sum(xi[:len(xi) - ig_num]) * 0.5

    def get_chisq_withweight_nosqrt(self, g, nu, weight, g_h, bins, bin_num2, inverse, ig_num):  # checked 2017-7-9!!!
        r"""
        to calculate the symmetry the shear estimators
        :param g: estimators from Fourier quad, 1-D numpy array
        :param nu: N + U for g1, N - U for g2, 1-D numpy array
        :param g_h: pseudo shear (guess)
        :param bins: bin of g for calculation of the symmetry, 1-D numpy array
        :param ig_num: the number of inner grid of bin to be neglected
        :return: chi square
        """
        G_h = g - nu * g_h
        num = numpy.histogram(G_h, bins, weights=weight)[0]
        n1 = num[0:bin_num2][inverse]
        n2 = num[bin_num2:]

        xi = ((n1 -n2)**2) / (n1 + n2)
        return numpy.sum(xi[:len(xi) - ig_num]) * 0.5

    def get_chisq_new(self, g, nu, g_h, bins, bin_num2, inverse, ig_num, num_exp):  # checked 2017-7-9!!!
        r"""
        to calculate the symmetry the shear estimators
        :param g: estimators from Fourier quad, 1-D numpy array
        :param nu: N + U for g1, N - U for g2, 1-D numpy array
        :param g_h: pseudo shear (guess)
        :param bins: bin of g for calculation of the symmetry, 1-D numpy array
        :param ig_num: the number of inner grid of bin to be neglected
        :return: chi square
        """
        G_h = g - nu * g_h
        num = numpy.histogram(G_h, bins)[0]
        n1 = num[0:bin_num2][inverse]
        n2 = num[bin_num2:]
        xi = ((n1 - num_exp) ** 2 + (n2 - num_exp) ** 2) / (n1 + n2)
        return numpy.sum(xi[:len(xi) - ig_num]) * 0.5

    def get_chisq_new_e(self, g, nu, g_h, bins, bin_num2, inverse, ig_num, num_exp):  # checked 2017-7-9!!!
        r"""
        to calculate the symmetry the shear estimators
        :param g: estimators from Fourier quad, 1-D numpy array
        :param nu: N + U for g1, N - U for g2, 1-D numpy array
        :param g_h: pseudo shear (guess)
        :param bins: bin of g for calculation of the symmetry, 1-D numpy array
        :param ig_num: the number of inner grid of bin to be neglected
        :return: chi square
        """
        G_h = g - nu * g_h
        num = numpy.histogram(G_h, bins)[0]
        n1 = num[0:bin_num2][inverse]
        n2 = num[bin_num2:]
        xi = ((n1 - num_exp) ** 2 + (n2 - num_exp) ** 2) / (num_exp*2)
        return numpy.sum(xi[:len(xi) - ig_num]) * 0.5

    def G_bin2d(self, mgs, mnus, g_corr, bins, resample=1, ig_nums=0):
        r"""
        to calculate the symmetry of two sets of shear estimators
        :param mgs: a two components list, two 1-D numpy arrays, contains two sets of
                    first shear estimators of Fourier quad ,
        :param mnus: a two components list, two 1-D numpy arrays, contains two sets of
                    shear estimators of Fourier quad (N,U), N + U for g1, N - U for g2
        :param g_corr: float, the correlation between the two sets of shear
        :param bins: a two components list, two 1-D numpy arrays, contains two bins for
                    the shear estimator
        :param resample: repeat the calculation for one shear PDF to reduct the noise
        :param ig_nums: a two components list, [num1, num2], the numbers of the inner grid
                        of each bin to be neglected (developing)
        :return: chi square
        """
        # half of the bin number
        ny, nx = int((len(bins[0]) - 1)/2), int((len(bins[1]) - 1)/2)
        chi_sq = 0
        mu = [0, 0]
        cov = [[abs(2*g_corr), g_corr],
               [g_corr, abs(2*g_corr)]]
        data_len = len(mgs[0])
        for i in range(resample):

            g_distri = numpy.random.multivariate_normal(mu,cov,data_len)

            mg1 = mgs[0] - mnus[0]*g_distri[:,0]
            mg2 = mgs[1] - mnus[1]*g_distri[:,1]
            num_arr = numpy.histogram2d(mg1, mg2, bins)[0]
            # | arr_1 | arr_2 |
            # | arr_3 | arr_4 |
            # chi square = 0.2*SUM[(arr_2 + arr_3 - arr_1 - arr_4)**2/(arr_1 + arr_2 + arr_3 + arr_4)]
            arr_1 = num_arr[0:ny, 0:nx][:,range(ny-1,-1,-1)]
            arr_2 = num_arr[0:ny, nx:2*nx]
            arr_3 = num_arr[ny:2*ny, 0:nx][range(ny-1,-1,-1)][:,range(nx-1,-1,-1)]
            arr_4 = num_arr[ny:2*ny, nx:2*nx][range(ny-1,-1,-1)]
            chi_sq += 0.5 * numpy.sum(((arr_2 + arr_3 - arr_1 - arr_4) ** 2) / (arr_1 + arr_2 + arr_3 + arr_4))

        return chi_sq/resample

    def fmin_g2d(self, mgs, mnus, bin_num, direct=True, ig_nums=0, left=-0.0015, right=0.0015,pic_path=False):
        r"""
        to find the true correlation between two sets of galaxies
        :param mgs: a two components list, two 1-D numpy arrays, contains two sets of
                    first shear estimators of Fourier quad ,
        :param mnus: a two components list, two 1-D numpy arrays, contains two sets of
                    shear estimators of Fourier quad (N,U), N + U for g1, N - U for g2
        :param bin_num: [num1, num2] for [mg1, mg2]
        :param direct: if true: fitting on a range of correlation guess,
                                it's a test for the large sample,
                                e.g. the CFHTLenS, too many pairs to do it in the traditional way.
        :param ig_nums: a two components list, [num1, num2], the numbers of the inner grid
                        of each bin to be neglected
        :param pic_path: the path for saving the chi square picture
        :param left, right: initial guess of the correlation,
                            0.002 is quite large (safe) for weak lensing
        :return: the correlation between mg1 and mg2
        """
        bins = [self.set_bin(mgs[0], bin_num), self.set_bin(mgs[1], bin_num)]
        if direct:
            # the initial guess of correlation value, 0.002, is quite large for weak lensing
            # so, it is safe
            fit_num = 6
            interval = (right - left) / 2
            # it's safe to fit a curve when the right -left < 0.0004
            while not interval < 0.0003:
                fit_range = numpy.linspace(left, right, fit_num+1)
                chi_sq = [self.G_bin2d(mgs, mnus, fit_range[i], bins, ig_nums=ig_nums) for i in range(fit_num+1)]
                cor_min = fit_range[chi_sq.index(min(chi_sq))]
                interval = (right - left) / 2
                left = max(left, cor_min - interval / 2)
                right = min(right, cor_min + interval / 2)
            fit_range = numpy.linspace(left, right, 21)

        else:
            iters = 0
            change = 1
            while change == 1:
                change = 0
                mc = (left + right) / 2.
                mcl = (mc + left) / 2.
                mcr = (mc + right) / 2.
                fmc = self.G_bin2d(mgs, mnus, mc, bins, ig_nums=ig_nums)
                fmcl = self.G_bin2d(mgs, mnus, mcl, bins, ig_nums=ig_nums)
                fmcr = self.G_bin2d(mgs, mnus, mcr, bins, ig_nums=ig_nums)
                temp = fmc + 30
                if fmcl > temp:
                    left = (mc + mcl) / 2.
                    change = 1
                if fmcr > temp:
                    right = (mc + mcr) / 2.
                    change = 1
                iters += 1
                if iters > 12:
                    break
            fit_range = numpy.linspace(left, right, 21)
        chi_sq = [self.G_bin2d(mgs, mnus, fit_range[i], bins, ig_nums=ig_nums) for i in range(len(fit_range))]
        coeff = tool_box.fit_1d(fit_range, chi_sq, 2, "scipy")
        corr_sig = numpy.sqrt(1 / 2. / coeff[2])
        g_corr = -coeff[1] / 2. / coeff[2]
        if pic_path:
            plt.scatter(fit_range,chi_sq)
            plt.title("$10^{4}$CORR: %.2f (%.4f)"%(g_corr*10000, corr_sig*10000))
            plt.plot(fit_range, coeff[0]+coeff[1]*fit_range+coeff[2]*fit_range**2)
            plt.xlim(left - 0.1*(right-left), right + 0.1*(right-left))
            plt.savefig(pic_path)
            plt.close()
            # plt.show()
        return -g_corr, corr_sig

    def find_shear_bk(self, g, nu, bin_num, ig_num=0, scale=1.1, left=-0.2, right=0.2,fit_num=60,chi_gap=40,fig_ax=False):  # checked 2017-7-9!!!
        """
        G1 (G2): the shear estimator for g1 (g2),
        N: shear estimator corresponding to the PSF correction
        U: the term for PDF-SYM
        V: the term for transformation
        :param g: G1 or G2, 1-D numpy arrays, the shear estimators of Fourier quad
        :param nu: N+U: for g1, N-U: for g2
        :param bin_num:
        :param ig_num:
        :param pic_path:
        :param left, right: the initial guess of shear
        :return: estimated shear and sigma
        """
        bins = self.set_bin(g, bin_num,scale)
        bin_num2 = int(bin_num * 0.5)
        inverse = range(int(bin_num / 2 - 1), -1, -1)
        same = 0
        iters = 0
        # m1 chi square & left & left chi square & right & right chi square
        records = numpy.zeros((15, 5))
        while True:
            templ = left
            tempr = right
            m1 = (left + right) / 2.
            m2 = (m1 + left) / 2.
            m3 = (m1 + right) / 2.
            fL = self.get_chisq(g, nu, left, bins, bin_num2, inverse, ig_num)
            fR = self.get_chisq(g, nu, right, bins, bin_num2, inverse, ig_num)
            fm1 = self.get_chisq(g, nu, m1, bins, bin_num2, inverse, ig_num)
            fm2 = self.get_chisq(g, nu, m2, bins, bin_num2, inverse, ig_num)
            fm3 = self.get_chisq(g, nu, m3, bins, bin_num2, inverse, ig_num)
            values = [fL, fm2, fm1, fm3, fR]
            points = [left, m2, m1, m3, right]
            records[iters, ] = fm1, left, fL, right, fR
            if max(values) < chi_gap:
                temp_left = left
                temp_right = right
            if fL > max(fm1, fm2, fm3) and fR > max(fm1, fm2, fm3):
                if fm1 == fm2:
                    left = m2
                    right = m1
                elif fm1 == fm3:
                    left = m1
                    right = m3
                elif fm2 == fm3:
                    left = m2
                    right = m3
                elif fm1 < fm2 and fm1 < fm3:
                    left = m2
                    right = m3
                elif fm2 < fm1 and fm2 < fm3:
                    right = m1
                elif fm3 < fm1 and fm3 < fm2:
                    left = m1
            elif fR > fm3 >= fL:
                if fL == fm3:
                    right = m3
                elif fL == fm1:
                    right = m1
                elif fL == fm2:
                    right = m2
                elif fm1 == fm2:
                    right = right
                elif fm1 < fm2 and fm1 < fL:
                    left = m2
                    right = m3
                elif fm2 < fL and fm2 < fm1:
                    right = m1
                elif fL < fm1 and fL < fm2:
                    right = m2
            elif fL > fm2 >= fR:
                if fR == fm2:
                    left = m2
                elif fR == fm1:
                    left = m1
                elif fR == fm3:
                    left = m3
                elif fm1 < fR and fm1 < fm3:
                    left = m2
                    right = m3
                elif fm3 < fm1 and fm3 < fR:
                    left = m1
                elif fR < fm1 and fR < fm3:
                    left = m3
                elif fm1 == fm3:
                    left = m1
                    right = m3

            if abs(left-right) < 1.e-5:
                g_h = (left+right)/2.
                break
            iters += 1
            if left == templ and right == tempr:
                same += 1
            if iters > 12 and same > 2 or iters > 14:
                g_h = (left+right)/2.
                break
                # print(left,right,abs(left-right))
        # fitting
        left_x2 = numpy.min(numpy.abs(records[:iters, 2] - fm1 - chi_gap))
        label_l = numpy.where(left_x2 == numpy.abs(records[:iters, 2] - fm1 - chi_gap))[0]
        if len(label_l > 1):
            label_l = label_l[0]

        right_x2 = numpy.min(numpy.abs(records[:iters, 4] - fm1 - chi_gap))
        label_r = numpy.where(right_x2 == numpy.abs(records[:iters, 4] - fm1 - chi_gap))[0]
        if len(label_r > 1):
            label_r = label_r[0]

        if left_x2 > right_x2:
            right = records[label_l, 3]
            left = 2*m1 - right
        else:
            left = records[label_r, 1]
            right = 2*m1 - left

        fit_range = numpy.linspace(left, right, fit_num)
        chi_sq = numpy.array([self.get_chisq(g, nu, g_hat, bins, bin_num2, inverse, ig_num) for g_hat in fit_range])

        coeff = tool_box.fit_1d(fit_range, chi_sq, 2, "scipy")

        # y = a1 + a2*x a3*x^2 = a3(x+a2/2/a3)^2 +...
        # gh = - a2/2/a3, gh_sig = \sqrt(1/2/a3)
        # g_h = -coeff[1] / 2. / coeff[2]
        g_sig = 0.70710678118/numpy.sqrt(coeff[2])

        if fig_ax:
            fig_ax.scatter(fit_range, chi_sq, alpha=0.7, s=10,c="C1")
            fig_ax.plot(fit_range, coeff[0] + coeff[1] * fit_range + coeff[2] * fit_range ** 2, alpha=0.7)
            text_str = "Num: %d\n%.5f\n%.5fx\n%.5f$x^2$\ng=%.5f (%.5f)"%(len(g),coeff[0],coeff[1],coeff[2],g_h, g_sig)
            fig_ax.text(0.1, 0.7, text_str, color='C3', ha='left', va='center', transform=fig_ax.transAxes,
                        fontsize=15)

        return g_h, g_sig, coeff

    def get_chisq_range(self, G, NU, bin_num, signal_guess, ig_num=0, scale=1.1, limited=None):
        bins = self.set_bin(G, bin_num, scale)

        bin_num2 = int(bin_num * 0.5)
        inverse = range(int(bin_num / 2 - 1), -1, -1)
        chi_sq = numpy.array([self.get_chisq(G, NU, g_hat, bins, bin_num2, inverse, ig_num) for g_hat in signal_guess])
        if limited:
            idx = chi_sq <= limited
            return signal_guess[idx], chi_sq[idx]
        else:
            return signal_guess, chi_sq

    def find_shear(self, g, nu, bin_num, ig_num=0, scale=1.1, left=-0.15, right=0.15, fit_num=60, chi_gap=40, fig_ax=False,loc_fit=False):
        """
        G1 (G2): the shear estimator for g1 (g2),
        N: shear estimator corresponding to the PSF correction
        U: the term for PDF-SYM
        V: the term for transformation
        :param g: G1 or G2, 1-D numpy arrays, the shear estimators of Fourier quad
        :param nu: N+U: for g1, N-U: for g2
        :param bin_num:
        :param ig_num:
        :param pic_path:
        :param left, right: the initial guess of shear
        :return: estimated shear and sigma
        """
        if len(g)==0:
            return 0,0,0
        else:
            bins = self.set_bin(g, bin_num, scale)

            bin_num2 = int(bin_num * 0.5)
            inverse = range(int(bin_num / 2 - 1), -1, -1)

            iters = 0
            change = 1
            while change == 1:
                change = 0
                mc = (left + right) / 2.
                mcl = left
                mcr = right
                fmc = self.get_chisq(g, nu, mc, bins, bin_num2, inverse, ig_num)
                fmcl = self.get_chisq(g, nu, mcl, bins, bin_num2, inverse, ig_num)
                fmcr = self.get_chisq(g, nu, mcr, bins, bin_num2, inverse, ig_num)
                temp = fmc + chi_gap

                if fmcl > temp:
                    left = (mc + mcl) / 2.
                    change = 1
                if fmcr > temp:
                    right = (mc + mcr) / 2.
                    change = 1
                iters += 1
                if iters > 12:
                    break

            fit_range = numpy.linspace(left, right, fit_num)
            chi_sq = numpy.array([self.get_chisq(g, nu, g_hat, bins, bin_num2, inverse, ig_num) for g_hat in fit_range])
            if fig_ax:
                fig_ax.scatter(fit_range, chi_sq, alpha=0.7, s=5,c="k")
            if loc_fit:
                min_tag = numpy.where(chi_sq == chi_sq.min())[0][0]
                chi_sq = chi_sq[min_tag - loc_fit: min_tag+loc_fit]
                fit_range = fit_range[min_tag - loc_fit: min_tag+loc_fit]

            coeff = tool_box.fit_1d(fit_range, chi_sq, 2, "scipy")

            # y = a1 + a2*x + a3*x^2 = a3(x+a2/2/a3)^2 +...
            # gh = - a2/2/a3, gh_sig = 1/ sqrt(1/2/a3)
            g_h = -coeff[1] / 2. / coeff[2]
            g_sig = 0.70710678118 / numpy.sqrt(coeff[2])

            if fig_ax:
                fig_ax.scatter(fit_range, chi_sq, alpha=0.7, s=10,c="C1")
                fig_ax.plot(fit_range, coeff[0] + coeff[1] * fit_range + coeff[2] * fit_range ** 2, alpha=0.7)
                text_str = "Num: %d\n%.5f\n%.5fx\n%.5f$x^2$\ng=%.5f (%.5f)"%(len(g),coeff[0],coeff[1],coeff[2],g_h, g_sig)
                fig_ax.text(0.1, 0.85, text_str, color='C3', ha='left', va='center', transform=fig_ax.transAxes,
                            fontsize=15)

            return g_h, g_sig, coeff

    def find_shear_withweight(self, g, nu, bin_num, ig_num=0, scale=1.1, left=-0.1, right=0.1, fit_num=60, chi_gap=40, fig_ax=False,loc_fit=False,w=None):
        """
        G1 (G2): the shear estimator for g1 (g2),
        N: shear estimator corresponding to the PSF correction
        U: the term for PDF-SYM
        V: the term for transformation
        :param g: G1 or G2, 1-D numpy arrays, the shear estimators of Fourier quad
        :param nu: N+U: for g1, N-U: for g2
        :param bin_num:
        :param ig_num:
        :param pic_path:
        :param left, right: the initial guess of shear
        :return: estimated shear and sigma
        """
        bins = self.set_bin(g, bin_num, scale)

        bin_num2 = int(bin_num * 0.5)
        inverse = range(int(bin_num / 2 - 1), -1, -1)

        iters = 0
        change = 1
        while change == 1:
            change = 0
            mc = (left + right) / 2.
            mcl = left
            mcr = right
            fmc = self.get_chisq_withweight(g, nu,w ,mc, bins, bin_num2, inverse, ig_num)
            fmcl = self.get_chisq_withweight(g, nu,w ,mcl, bins, bin_num2, inverse, ig_num)
            fmcr = self.get_chisq_withweight(g, nu,w, mcr, bins, bin_num2, inverse, ig_num)
            temp = fmc + chi_gap

            if fmcl > temp:
                left = (mc + mcl) / 2.
                change = 1
            if fmcr > temp:
                right = (mc + mcr) / 2.
                change = 1
            iters += 1
            if iters > 12:
                break

        fit_range = numpy.linspace(left, right, fit_num)
        chi_sq = numpy.array([self.get_chisq_withweight(g, nu,w, g_hat, bins, bin_num2, inverse, ig_num) for g_hat in fit_range])
        if fig_ax:
            fig_ax.scatter(fit_range, chi_sq, alpha=0.7, s=5,c="k")
        if loc_fit:
            min_tag = numpy.where(chi_sq == chi_sq.min())[0][0]
            chi_sq = chi_sq[min_tag - loc_fit: min_tag+loc_fit]
            fit_range = fit_range[min_tag - loc_fit: min_tag+loc_fit]

        coeff = tool_box.fit_1d(fit_range, chi_sq, 2, "scipy")

        # y = a1 + a2*x + a3*x^2 = a3(x+a2/2/a3)^2 +...
        # gh = - a2/2/a3, gh_sig = 1/ sqrt(1/2/a3)
        g_h = -coeff[1] / 2. / coeff[2]
        g_sig = 0.70710678118 / numpy.sqrt(coeff[2])

        if fig_ax:
            fig_ax.scatter(fit_range, chi_sq, alpha=0.7, s=10,c="C1")
            fig_ax.plot(fit_range, coeff[0] + coeff[1] * fit_range + coeff[2] * fit_range ** 2, alpha=0.7)
            text_str = "Num: %d\n%.5f\n%.5fx\n%.5f$x^2$\ng=%.5f (%.5f)"%(len(g),coeff[0],coeff[1],coeff[2],g_h, g_sig)
            fig_ax.text(0.1, 0.85, text_str, color='C3', ha='left', va='center', transform=fig_ax.transAxes,
                        fontsize=15)
        return g_h, g_sig, coeff

    def find_shear_withweight_nosqrt(self, g, nu, bin_num, ig_num=0, scale=1.1, left=-0.1, right=0.1, fit_num=60, chi_gap=40, fig_ax=False,loc_fit=False,w=None):
        """
        G1 (G2): the shear estimator for g1 (g2),
        N: shear estimator corresponding to the PSF correction
        U: the term for PDF-SYM
        V: the term for transformation
        :param g: G1 or G2, 1-D numpy arrays, the shear estimators of Fourier quad
        :param nu: N+U: for g1, N-U: for g2
        :param bin_num:
        :param ig_num:
        :param pic_path:
        :param left, right: the initial guess of shear
        :return: estimated shear and sigma
        """
        bins = self.set_bin(g, bin_num, scale)

        bin_num2 = int(bin_num * 0.5)
        inverse = range(int(bin_num / 2 - 1), -1, -1)

        iters = 0
        change = 1
        while change == 1:
            change = 0
            mc = (left + right) / 2.
            mcl = left
            mcr = right
            fmc = self.get_chisq_withweight_nosqrt(g, nu,w ,mc, bins, bin_num2, inverse, ig_num)
            fmcl = self.get_chisq_withweight_nosqrt(g, nu,w ,mcl, bins, bin_num2, inverse, ig_num)
            fmcr = self.get_chisq_withweight_nosqrt(g, nu,w, mcr, bins, bin_num2, inverse, ig_num)
            temp = fmc + chi_gap

            if fmcl > temp:
                left = (mc + mcl) / 2.
                change = 1
            if fmcr > temp:
                right = (mc + mcr) / 2.
                change = 1
            iters += 1
            if iters > 12:
                break

        fit_range = numpy.linspace(left, right, fit_num)
        chi_sq = numpy.array([self.get_chisq_withweight_nosqrt(g, nu,w, g_hat, bins, bin_num2, inverse, ig_num) for g_hat in fit_range])
        if fig_ax:
            fig_ax.scatter(fit_range, chi_sq, alpha=0.7, s=5,c="k")
        if loc_fit:
            min_tag = numpy.where(chi_sq == chi_sq.min())[0][0]
            chi_sq = chi_sq[min_tag - loc_fit: min_tag+loc_fit]
            fit_range = fit_range[min_tag - loc_fit: min_tag+loc_fit]

        coeff = tool_box.fit_1d(fit_range, chi_sq, 2, "scipy")

        # y = a1 + a2*x + a3*x^2 = a3(x+a2/2/a3)^2 +...
        # gh = - a2/2/a3, gh_sig = 1/ sqrt(1/2/a3)
        g_h = -coeff[1] / 2. / coeff[2]
        g_sig = 0.70710678118 / numpy.sqrt(coeff[2])

        if fig_ax:
            fig_ax.scatter(fit_range, chi_sq, alpha=0.7, s=10,c="C1")
            fig_ax.plot(fit_range, coeff[0] + coeff[1] * fit_range + coeff[2] * fit_range ** 2, alpha=0.7)
            text_str = "Num: %d\n%.5f\n%.5fx\n%.5f$x^2$\ng=%.5f (%.5f)"%(len(g),coeff[0],coeff[1],coeff[2],g_h, g_sig)
            fig_ax.text(0.1, 0.85, text_str, color='C3', ha='left', va='center', transform=fig_ax.transAxes,
                        fontsize=15)
        return g_h, g_sig, coeff

    def find_shear_new(self, g, nu, bin_num, ig_num=0, scale=1.1, left=-0.2, right=0.2, fit_num=60,chi_gap=40,fig_ax=False):
        """
        G1 (G2): the shear estimator for g1 (g2),
        N: shear estimator corresponding to the PSF correction
        U: the term for PDF-SYM
        V: the term for transformation
        :param g: G1 or G2, 1-D numpy arrays, the shear estimators of Fourier quad
        :param nu: N+U: for g1, N-U: for g2
        :param bin_num:
        :param ig_num:
        :param pic_path:
        :param left, right: the initial guess of shear
        :return: estimated shear and sigma
        """
        bins = self.set_bin(g,bin_num,scale)

        bin_num2 = int(bin_num * 0.5)
        inverse = range(int(bin_num / 2 - 1), -1, -1)
        num_ini = numpy.histogram(g, bins)[0]

        n1 = num_ini[0:int(bin_num / 2)][inverse]
        n2 = num_ini[int(bin_num / 2):]
        num_exp = (n1 + n2) / 2

        iters = 0
        change = 1
        while change == 1:
            change = 0
            mc = (left + right) / 2.
            mcl = left
            mcr = right
            fmc = self.get_chisq_new(g, nu, mc, bins, bin_num2, inverse, ig_num, num_exp)
            fmcl = self.get_chisq_new(g, nu, mcl, bins, bin_num2, inverse, ig_num, num_exp)
            fmcr = self.get_chisq_new(g, nu, mcr, bins, bin_num2, inverse, ig_num, num_exp)
            temp = fmc + chi_gap

            if fmcl > temp:
                left = (mc + mcl)/2.
                change = 1
            if fmcr > temp:
                right = (mc + mcr)/2.
                change = 1
            iters += 1
            if iters > 12:
                break

        fit_range = numpy.linspace(left, right, fit_num)
        chi_sq = numpy.array([self.get_chisq_new(g, nu, g_hat, bins, bin_num2, inverse, ig_num, num_exp) for g_hat in fit_range])

        coeff = tool_box.fit_1d(fit_range, chi_sq, 2, "scipy")

        # y = a1 + a2*x a3*x^2 = a2(x+a1/2/a2)^2 +...
        # gh = - a1/2/a2, gh_sig = \sqrt(1/2/a2)
        g_h = -coeff[1] / 2. / coeff[2]
        g_sig = 0.70710678118/numpy.sqrt(coeff[2])

        if fig_ax:
            fig_ax.scatter(fit_range, chi_sq, alpha=0.7,s=5)
            fig_ax.plot(fit_range, coeff[0]+coeff[1]*fit_range+coeff[2]*fit_range**2,alpha=0.7)
            text_str = "Num: %d\n%.5f\n%.5fx\n%.5f$x^2$\ng=%.5f (%.5f)"%(len(g),coeff[0],coeff[1],coeff[2],g_h, g_sig)
            fig_ax.text(0.1, 0.85, text_str, color='C3', ha='left', va='center', transform=fig_ax.transAxes,
                        fontsize=15)

        return g_h, g_sig, coeff


    def find_shear_mean(self, G, N, weight=1):
        num = G.shape[0]
        g = numpy.sum(G)/numpy.sum(N)
        g_sig = numpy.sqrt(numpy.mean((G*weight)**2)/(numpy.mean(N*weight))**2)/numpy.sqrt(num)

        return g, g_sig



