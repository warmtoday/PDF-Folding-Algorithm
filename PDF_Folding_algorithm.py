import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import os
from mpi4py import MPI
from numpy import fft
from astropy.io import fits
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredSizeBar)
import tool_box
import h5py
import matplotlib
from Fourier_Quad import Fourier_Quad
from multiprocessing import Pool
from matplotlib import cm
import string
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import Planck15
import math
matplotlib.use('Agg')

def kstransform(g1map,g2map):

    gfs = np.fft.fft2(g1map+g2map*(1j))
    gf = np.fft.fftshift(gfs)
    g1eff = np.zeros([len(gf),len(gf)])
    g2eff = np.zeros([len(gf),len(gf)])

    r = int(len(gf)/2)
    for j in range(len(gf)):
        for i in range(len(gf)):
            x = i - r
            y = j - r
            if x == 0 and y == 0:
                x = 1
                y = 1
            g1eff[j,i] = (x**2-y**2)/(x**2+y**2)
            g2eff[j,i] = (2*x*y)/(x**2+y**2)
    temp = np.fft.ifftshift((g1eff-g2eff*(1j))*gf)
    kmap = np.real(np.fft.ifft2(temp))
    kmap_B = np.imag(np.fft.ifft2(temp))
    return kmap,kmap_B


dirs = os.listdir('/lustre/home/acct-phyzj/phyzj-m31/whz/DECALS_REC/hscmap/')#the path where data existing
filecount = 0
dirs0 = []

#files that named "DR3" in prefix and ".h5"in suffix.
for filename0 in dirs:
    if (filename0[0] == 'D') and (filename0[1] == 'R') and filename0[-1] == '5' and filename0[2] == '3':
        dirs0.append(filename0)

#reconstruct shear and kappa map of each small region. 
for filename in dirs0:
    if filename[0] == 'D' and (filename[1] == 'R') and filename[2] == '3' and filename[-1] == '5':
        filecount += 1
        np.random.seed(123)
        f1 = h5py.File(filename,'r')

        cluster_ra =float(filename.split('_')[1])
        cluster_dec = float(filename.split('_')[2])
        zl = float(filename.split('_')[3][:-3])

        zl = 0.1
        if (cluster_ra > 155) or (cluster_ra < 128) or (cluster_dec > 5) or (cluster_dec < -5): continue

        cluster_ra_l = int((cluster_ra - 0.25)*1000)/1000
        cluster_ra_ll = int((cluster_ra - 0.5)*1000)/1000

        cluster_ra_h = int((cluster_ra + 0.25)*1000)/1000
        cluster_ra_hh = int((cluster_ra + 0.5)*1000)/1000

        cluster_dec_l = int((cluster_dec - 0.25)*1000)/1000
        cluster_dec_ll = int((cluster_dec - 0.5)*1000)/1000

        cluster_dec_h = int((cluster_dec + 0.25)*1000)/1000
        cluster_dec_hh = int((cluster_dec + 0.5)*1000)/1000

        cluster_ra_12arcminl = int((cluster_ra-0.20)*1000)/1000
        cluster_ra_12arcminh = int((cluster_ra+0.20)*1000)/1000
        cluster_dec_12arcminl = int((cluster_dec-0.20)*1000)/1000
        cluster_dec_12arcminh = int((cluster_dec+0.20)*1000)/1000

        cluster_ra = int(cluster_ra*1000)/1000
        cluster_dec = int(cluster_dec*1000)/1000

        ra_ticks_halfdegree = [cluster_ra_l,cluster_ra,cluster_ra_h]
        dec_ticks_halfdegree = [cluster_dec_l,cluster_dec,cluster_dec_h]

        ra_ticks_degree = [cluster_ra_ll, cluster_ra, cluster_ra_hh]
        dec_ticks_degree = [cluster_dec_ll, cluster_dec, cluster_dec_hh]

        ra_ticks_24arcmin = [cluster_ra_12arcminl, cluster_ra, cluster_ra_12arcminh]
        dec_ticks_24arcmin = [cluster_dec_12arcminl, cluster_dec, cluster_dec_12arcminh]

        radius = 72  #Each region is in 1.2deg*1.2deg. We devide it into 72pixels*72pixels. 
        radius_x = 72
        radius_y = 72
        order = int(radius / 2)
        orders = range((order+1)*radius)
        scale_factor = 1
        rad = 0.6
        fq = Fourier_Quad(100,28)

        id_select = (np.abs(f1['rapart'][()] - cluster_ra) <= rad) & (
                np.abs(f1['decpart'][()] - cluster_dec) <= rad) &(f1['redshiftpart'][()]>zl-0.05)&(f1['redshiftpart'][()]<=zl+0.05)

        xpart = (f1['rapart'][()][id_select] - cluster_ra) * 60
        ypart = (f1['decpart'][()][id_select] - cluster_dec) * 60
        redshift = f1['redshiftpart'][()][id_select]

        total_num = len(xpart)
        res = 1

        if total_num < 10000: continue#if there are few galaxies, we skip this region.

        zl01 = zl + 0.2 #we suppose the background galaxies are in zl+0.2 with using "shear ratio". The "shear ratio" is defined as "s_cr".
        dist_lens = Distance(unit=u.Mpc, z=zl, cosmology=Planck15)
        dist_source = Distance(unit=u.Mpc, z=zl01, cosmology=Planck15)
        ang_lens_dist = dist_lens[()] / (1 + zl)
        ang_source_dist = dist_source[()] / (1 + zl01)
        s_cr1 = 5.57 * ang_source_dist / (
                ang_lens_dist * (ang_source_dist - ang_lens_dist)) / 100 ** 2 / 67.35 / 1000 / 1000

        dist_lens = Distance(unit=u.Mpc, z=zl, cosmology=Planck15)
        dist_source = Distance(unit=u.Mpc, z=redshift, cosmology=Planck15)
        ang_lens_dist = dist_lens[()] / (1 + zl)
        ang_source_dist = dist_source[()] / (1 + redshift)
        s_cr2 = 5.57 * ang_source_dist / (
                ang_lens_dist * (ang_source_dist - ang_lens_dist)) / 100 ** 2 / 67.35 / 1000 / 1000

        xpart = (f1['rapart'][()][id_select] - cluster_ra) * 60
        ypart = (f1['decpart'][()][id_select] - cluster_dec) * 60
        factor = np.ones(len(xpart))

        xrange = np.arange(int(-radius / 2), int(radius / 2))
        yrange = np.arange(int(-radius / 2), int(radius / 2))
        xarray, yarray = np.meshgrid(xrange, yrange)

        number_radius =1
        number_avr = 1

        density_map = np.ones([radius_x, radius_y])
        for i in range(radius_x):
            for j in range(radius_y):
                id = ((i - int(radius_x/2) - xpart)**2 + ((j - int(radius_y/2)) - ypart)**2<number_radius**2)
                if len(xpart[id]) > 0:
                    density_map[j, i] = np.sqrt(len(xpart[id]))

        density0 = np.ones(len(xpart))
        for c in range(len(xpart)):
            density0[c] = density_map[math.floor(ypart[c]+radius/2),math.floor(xpart[c]+radius/2)]


        g1map_temp = np.zeros([radius_x, radius_y])

        while (res > 0.1):
            if res > 1:break

            g1map = np.zeros([radius_x, radius_y])
            g2map = np.zeros([radius_x, radius_y])

            id_select = (np.abs(f1['rapart'][()] - cluster_ra) <= rad) & \
                        (np.abs(f1['decpart'][()] - cluster_dec) <= rad) &(f1['redshiftpart'][()]>zl-0.05)&(f1['redshiftpart'][()]<=zl+0.05)
            redshift = f1['redshiftpart'][()][id_select]

            zl01 = zl + 0.2
            dist_lens = Distance(unit=u.Mpc, z=zl, cosmology=Planck15)
            dist_source = Distance(unit=u.Mpc, z=zl01, cosmology=Planck15)
            ang_lens_dist = dist_lens[()] / (1 + zl)
            ang_source_dist = dist_source[()] / (1 + zl01)
            s_cr1 = 5.57 * ang_source_dist / (
                    ang_lens_dist * (ang_source_dist - ang_lens_dist)) / 100 ** 2 / 67.35 / 1000 / 1000

            dist_lens = Distance(unit=u.Mpc, z=zl, cosmology=Planck15)
            dist_source = Distance(unit=u.Mpc, z=zl01, cosmology=Planck15)
            ang_lens_dist = dist_lens[()] * 0.6735
            ang_source_dist = dist_source[()] * 0.6735
            s_cr1_scaled = ang_source_dist / (
                    ang_lens_dist * (ang_source_dist - ang_lens_dist)) * 1662895.2081868195 / (1 + zl)

            dist_lens = Distance(unit=u.Mpc, z=zl, cosmology=Planck15)
            dist_source = Distance(unit=u.Mpc, z=redshift, cosmology=Planck15)
            ang_lens_dist = dist_lens[()] / (1 + zl)
            ang_source_dist = dist_source[()] / (1 + redshift)
            s_cr2 = 5.57 * ang_source_dist / (
                    ang_lens_dist * (ang_source_dist - ang_lens_dist)) / 100 ** 2 / 67.35 / 1000 / 1000

            trans_factor = (s_cr1 / s_cr2)
            s_cr = (s_cr2 / s_cr1)

            #recovering coefficient in front of each mode in "orders". Different order has different "i,j".
            for t in orders:
                    i = int(t / radius)
                    j = int(t % radius - (radius / 2) + 1)

                    if i == 0 and j < 0: continue
                    if i == order and j < 0: continue

                    # gaussfun = np.exp(-(j ** 2 + i ** 2) / (2 * (order / 6) ** 2)) * (np.exp(1 / 2))
                    gaussfun = 1
                    print('%s,%s'%(j,i))

                    xpart = (f1['rapart'][()][id_select]-cluster_ra) * 60
                    ypart = (f1['decpart'][()][id_select]-cluster_dec) * 60
                    G1part = f1['G1part'][()][id_select]*s_cr
                    G2part = f1['G2part'][()][id_select]*s_cr
                    Npart = f1['Npart'][()][id_select]
                    Upart = -f1['Upart'][()][id_select]
                    density = density0

                    # cos term
                    idcc0 = np.cos(i*2*np.pi/radius_x*xpart+j*2*np.pi/radius_y*ypart)!=0

                    idcc = np.cos(i*2*np.pi/radius_x*xpart+j*2*np.pi/radius_y*ypart)<0
                    tmpcc = np.cos(i*2*np.pi/radius_x*xpart+j*2*np.pi/radius_y*ypart)

                    weight = np.abs(np.cos(i*2*np.pi/radius_x*xpart+j*2*np.pi/radius_y*ypart))/density/s_cr
                    tp = np.cos(i*2*np.pi/radius_x*xpart+j*2*np.pi/radius_y*ypart)
                    tmpcc[idcc] = -tp[idcc]
                    tmpcc /= density
                    G1cc = G1part
                    G2cc = G2part

                    G1cc[idcc] = -G1part[idcc]
                    G2cc[idcc] = -G2part[idcc]
                    g_cc,cc_sig = fq.find_shear_withweight(G1cc,tmpcc/factor*(Npart+Upart),bin_num=2,w=weight)[:2]
                    g2_cc, cc_sig2 = fq.find_shear_withweight(G2cc, tmpcc/factor * (Npart - Upart),
                                                            bin_num=2, w=weight)[:2]

                    if cc_sig!=cc_sig:
                        cc_sig = 0.1
                        g_cc =0
                    if cc_sig2!=cc_sig2:
                        cc_sig2=0.1
                        g2_cc = 0

                    print("g1_c%sc%s:%s,%s"%(i,j,g_cc,cc_sig))
                    print("g2_c%sc%s:%s,%s"%(i,j,g2_cc,cc_sig2))

                    G1part = f1['G1part'][()][id_select]*s_cr
                    G2part = f1['G2part'][()][id_select]*s_cr
                    Npart = f1['Npart'][()][id_select]
                    Upart = -f1['Upart'][()][id_select]

                    # sin term
                    idss0 = np.sin(i * 2 * np.pi / radius_x * xpart+j * 2 * np.pi / radius_y * ypart) != 0
                    if (i == 0 and j == 0) or (i == radius / 2 and j == 0) or (i == 0 and j == radius / 2) or (
                            i == radius / 2 and j == radius / 2):
                        g_ss = 0
                        g2_ss = 0
                        ss_sig = 0
                        ss_sig2 = 0
                    else:
                        idss = np.sin(i * 2 * np.pi / radius_x * xpart+j * 2 * np.pi / radius_y * ypart) < 0
                        tmpss = np.sin(i * 2 * np.pi / radius_x * xpart+j * 2 * np.pi / radius_y * ypart)
                        weight = np.abs(np.sin(i * 2 * np.pi / radius_x * xpart+j * 2 * np.pi / radius_y * ypart))/density/s_cr
                        tp = np.sin(i * 2 * np.pi / radius_x * xpart+j * 2 * np.pi / radius_y * ypart)
                        tmpss[idss] = -tp[idss]
                        tmpss /= density
                        G1ss = G1part
                        G2ss = G2part
                        G1ss[idss] = -G1part[idss]
                        G2ss[idss] = -G2part[idss]
                        g_ss, ss_sig = fq.find_shear_withweight(G1ss, tmpss/factor * (Npart + Upart), bin_num=2,w=weight)[:2]
                        g2_ss,ss_sig2 = fq.find_shear_withweight(G2ss, tmpss/factor * (Npart - Upart), bin_num=2,w=weight)[:2]

                    if ss_sig != ss_sig:
                        ss_sig = 0.1
                        g_ss = 0
                    if ss_sig2!=ss_sig2:
                        ss_sig2 = 0.1
                        g2_ss = 0

                    print("g1_s%ss%s:%s,%s" % (i, j, g_ss, ss_sig))
                    print("g2_s%ss%s:%s,%s" % (i, j, g2_ss, ss_sig2))

                    G1part = f1['G1part'][()][id_select]*s_cr
                    G2part = f1['G2part'][()][id_select]*s_cr
                    Npart = f1['Npart'][()][id_select]
                    Upart = -f1['Upart'][()][id_select]

                    g1map += gaussfun*(g_cc * (np.cos(i * 2 * np.pi / radius_x * xarray+j * 2 * np.pi / radius_y * yarray)) \
                             + g_ss * (np.sin(i * 2 * np.pi / radius_x * xarray+j * 2 * np.pi / radius_y * yarray)))

                    g2map += gaussfun*(g2_cc * (np.cos(i * 2 * np.pi / radius_x * xarray+j * 2 * np.pi / radius_y * yarray)) \
                             + g2_ss * (np.sin(i * 2 * np.pi / radius_x * xarray+j * 2 * np.pi / radius_y * yarray)))


            g1map0 = g1map / density_map
            g2map0 = g2map / density_map
            kmap, kmap_B = kstransform(g1map0, g2map0)

        Sigma_field = kmap * s_cr1_scaled
        SigmaB_field = kmap_B * s_cr1_scaled

        f2 = h5py.File('PF_%s_%s_%s.h5'%(cluster_ra,cluster_dec,zl),'w')
        f2['k'] = kmap*1/scale_factor
        f2['kb'] = kmap_B*1/scale_factor
        f2['g1map'] = g1map0
        f2['g2map'] = g2map0
        f2['density_map'] = density_map
        f2['Sigma_field'] = Sigma_field
        f2['SigmaB_field'] = SigmaB_field
        f2.close()
