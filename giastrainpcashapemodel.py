#!/usr/bin/env python
"""
FILE: giastrainpcashapemodel.py
LAST MODIFIED: 19/03/18
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
        n = np.arange(1,len(x)+1)
        _out = np.hstack([n[:,np.newaxis], x])
        np.savetxt(
            file, _out, delimiter=', ',
            fmt=['%8d', '%10.6f', '%10.6f', '%10.6f'],
            header=header
            )
    else:
        writer = vtktools.Writer(v=x.v, f=x.f)
        writer.write(file)

def do_pca(args):
    paths = np.loadtxt(args.path_file, dtype=str, delimiter=',')
    model_ext = path.splitext(paths[0])[1]

    # read input models
    models = []
    X = []

    # just load 1st if pc precomputed. 
    if args.pc_path is not None:
        paths = paths[:1]

    print('loading {} input datasets'.format(len(paths)))
    for in_path in paths:
        if args.points_only:
            x = np.loadtxt(in_path, delimiter=',', skiprows=1, usecols=(1,2,3))
            models.append(x)
        else:
            model = vtktools.loadpoly(in_path)
            x = model.v
            models.append(model)
        X.append(x.ravel())

    X = np.array(X).T

    # do pca
    if args.pc_path is None:
        if len(X.shape) != 2:
            raise RuntimeError('Input data shape must be 2D, but is instead {}'.format(X.shape))

        pca = PCA.PCA()
        pca.setData(X)
        print('data shape: {}'.format(X.shape))

        if args.ncomponents:
            pca.inc_svd_decompose(args.ncomponents)
        else:
            pca.svd_decompose()

        pc = pca.PC
    else:
        pc = PCA.loadPrincipalComponents(args.pc_path)

    mean_model = recon_model(pc.mean, args.points_only, models[0])
    
    # save pca model if we didnt preload
    if args.out:
        if args.pc_path is None:
            pc.save(args.out)
            pc.savemat(args.out)
            save_model(
                mean_model, args.out+'_mean'+model_ext, args.points_only,
                'mean model'
                )

    # reconstruct modal models
    recon_models = []
    if args.recon_modes is not None:
        for mi in args.recon_modes:
            xr1, xr2 = recon_data(pc, mi, 2.0)
            mr1 = recon_model(xr1, args.points_only, models[0])
            mr2 = recon_model(xr2, args.points_only, models[0])
            save_model(
                mr1, args.out+'_recon_pc{}{}'.format(mi, 'p2')+model_ext,
                args.points_only, 'recon pc {} {}'.format(mi, '+2sd')
                )
            save_model(
                mr2, args.out+'_recon_pc{}{}'.format(mi, 'm2')+model_ext,
                args.points_only, 'recon pc {} {}'.format(mi, '-2sd')
                )
            recon_models.append((mr1, mr2))

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
        PCA.plotModeScatter(pc, title='Shape space (modes 0 and 1)', pointLabels=[str(i) for i in range(len(models))], nTailLabels=5)

        try:
            from gias2.visualisation import fieldvi
            has_mayavi = True
        except ImportError:
            has_mayavi = False

        if has_mayavi:
            v = fieldvi.Fieldvi()
            v.addPC('principal components', pc)
            if args.points_only:
                v.addData(
                    'mean', mean_model,
                    renderArgs={'mode':'point', 'color':tuple(args.colour)}
                    )
            else:
                v.addTri(
                    'mean', mean_model,
                    renderArgs={'color':tuple(args.colour)}
                    )

            recon_opa = 0.5
            if args.recon_modes is not None:
                for mi in args.recon_modes:
                    if args.points_only:
                        v.addData(
                            'pc{} +2sd'.format(mi), recon_models[mi][0],
                            renderArgs={'mode':'point', 'color':tuple(args.recon_colour_1), 'opacity':args.recon_opacity}
                            )
                        v.addData(
                            'pc{} -2sd'.format(mi), recon_models[mi][1],
                            renderArgs={'mode':'point', 'color':tuple(args.recon_colour_2), 'opacity':args.recon_opacity}
                            )
                    else:
                        v.addTri(
                            'pc{} +2sd'.format(mi), recon_models[mi][0],
                            renderArgs={'color':tuple(args.recon_colour_1), 'opacity':args.recon_opacity}
                            )
                        v.addTri(
                            'pc{} -2sd'.format(mi), recon_models[mi][1],
                            renderArgs={'color':tuple(args.recon_colour_2), 'opacity':args.recon_opacity}
                            )

            v.scene.background=tuple(args.bgcolour)
            v.start()
            
            if sys.version_info.major==2:
                ret = raw_input('press any key and enter to exit')
            else:
                ret = input('press any key and enter to exit')
        else:
            print('Visualisation error: cannot import mayavi')

    return pc

#=============================================================================#
def main():
    parser = argparse.ArgumentParser(
        description='Run PCA on a set of shapes.')
    parser.add_argument(
        'path_file',
        help='text file containing the paths of input models.'
        )
    parser.add_argument(
        '-n', '--ncomponents',
        default=None,
        type=int,
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
        '-c', '--colour',
        nargs=3, type=float, default=[0.85, 0.8, 0.5],
        help='Colour of the main model. 3 values between 0 and 1 representing RGB values.'
        )
    parser.add_argument(
        '--recon-colour-1',
        nargs=3, type=float, default=[0.9, 0.5, 0.5],
        help='Colour of the main model. 3 values between 0 and 1 representing RGB values.'
        )
    parser.add_argument(
        '--recon-colour-2',
        nargs=3, type=float, default=[0.5, 0.5, 0.9],
        help='Colour of the main model. 3 values between 0 and 1 representing RGB values.'
        )
    parser.add_argument(
        '--bgcolour',
        nargs=3, type=float, default=[1.0, 1.0, 1.0],
        help='Colour of the background. 3 values between 0 and 1 representing RGB values.'
        )
    parser.add_argument(
        '--recon-opacity',
        type=float, default=0.5,
        help='Opacity of reconstructions.'
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

    pc = do_pca(args)

if __name__ == '__main__':
    main()