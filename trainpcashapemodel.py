"""
FILE: trainpcashapemodel.py
LAST MODIFIED: 24/05/17
DESCRIPTION:
Script for performing PCA on a set of shapes.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

from os import path
import sys
import argparse
import copy

import numpy as np
from gias2.learning import PCA
from gias2.mesh import vtktools

def recon_data(pc, mode, sd):
    x1 = pc.reconstruct(pc.getWeightsBySD([mode], [ sd]), [mode])
    x2 = pc.reconstruct(pc.getWeightsBySD([mode], [-sd]), [mode])
    return x1, x2

def recon_model(x, points_only, templatemesh=None):
    if points_only:
        return x.reshape((-1,3))
    else:
        m = copy.deepcopy(templatemesh)
        m.v = x.reshape((-1,3))
        return m

def save_model(x, file, points_only, header=None):

    if points_only:
        np.savetxt(
            file, x, delimiter=',',
            fmt=['%10.6f', '%10.6f', '%10.6f'],
            )
    else:
        writer = vtktools.Writer(v=x.v, f=x.f)
        writer.write(file)

def main(args):
    paths = np.loadtxt(args.path_file, dtype=str, delimiter=',')
    model_ext = path.splitext(paths[0])[1]

    # read input models
    models = []
    X = []

    # just load 1st if pc precomputed. 
    if args.pc_path is not None:
        paths = paths[:1]

    for in_path in paths:
        if args.points_only:
            x = np.loadtxt(in_path, delimiter=',', skiprows=1)
            models.append(x)
        else:
            model = vtktools.loadpoly(in_path)
            x = model.v
            models.append(model)
        X.append(x.ravel())

    X = np.array(X).T

    # do pca
    if args.pc_path is None:
        pca = PCA.PCA()
        pca.setData(X)
        print('data shape: {}'.format(X.shape))
        # pca.inc_svd_decompose(args.ncomponents)
        pca.svd_decompose()
        pc = pca.PC
    else:
        pc = PCA.loadPrincipalComponents(args.pc_path)

    mean_model = recon_model(pc.mean, args.points_only, models[0])
    
    # save pca model if we didnt preload
    if args.out:
        if args.pc_path is None:
            pc.save(args.out)
            save_model(
                mean_model, args.out+'_mean'+model_ext, args.points_only,
                'mean model'
                )

    # reconstruct modal models
    if args.recon_modes is not None:
        for mi in args.recon_modes:
            xr1, xr2 = recon_data(pc, mi, 2.0)
            mr1 = recon_model(xr1, args.points_only, models[0])
            mr2 = recon_model(xr2, args.points_only, models[0])
            save_model(mr1, args.out+'_recon_pc{}{}'.format(mi, 'p2')+model_ext, args.points_only)
            save_model(mr2, args.out+'_recon_pc{}{}'.format(mi, 'm2')+model_ext, args.points_only)

    # visualise
    componentVar = pc.getNormSpectrum()
    print('PC Percentage Significance')
    for i in range(args.plot_pcs):
        try:
            print('pc%d: %4.2f%%'%(i+1, componentVar[i]*100))
        except IndexError:
            pass

    if args.view:
        PCA.plotSpectrum(pc, args.plot_pcs, title='Mode Variance')
        PCA.plotModeScatter(pc, title='Shape space (modes 0 and 1)')

        try:
            from gias2.visualisation import fieldvi
            has_mayavi = True
        except ImportError:
            has_mayavi = False

        if has_mayavi:
            v = fieldvi.Fieldvi()
            v.addPC('principal components', pc)
            if args.points_only:
                v.addData('mean', mean_model, renderArgs={'mode':'point', 'color':(1,0,0)})
            else:
                v.addTri('mean', mean_model, renderArgs={'color':(1,0,0)})

            v.scene.background=(0,0,0)
            v.configure_traits()
        else:
            print('Visualisation error: cannot import mayavi')

#=============================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run PCA on a set of shapes.')
    parser.add_argument(
        'path_file',
        help='text file containing the paths of input models.'
        )
    parser.add_argument(
        '-n', '--ncomponents',
        default=10,
        help='Number of PCs to find if using incremental PCA.'
        )
    parser.add_argument(
        '-p', '--pc-path',
        help='.pc.npz file containing a precomputed shape model.'
        )
    parser.add_argument(
        '--points-only',
        action='store_true',
        help='Model are point clouds only. Expected file format is 1 header line, then n,x,y,z on each line after'
        )
    parser.add_argument(
        '-r', '--recon_modes',
        nargs='+',
        type=int,
        help='Reconstruct models at +2 sd along the given modes, starting from 0. Saves to file.'
        )
    parser.add_argument(
        '-o', '--out',
        help='file path of the output pca file.'
        )
    parser.add_argument(
        '-v', '--view',
        action='store_true',
        help='Visualise results.'
        )
    parser.add_argument(
        '--plot_pcs',
        type=int,
        default=10,
        help='First n modes to plot.'
        )
    args = parser.parse_args()

    main(args)