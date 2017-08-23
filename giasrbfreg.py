#!/usr/bin/env python
"""
FILE: rbfreg.py
LAST MODIFIED: 23/05/17
DESCRIPTION:
Script for registering one model to another using RBFs.

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
import numpy as np
from scipy.spatial import cKDTree
import copy
import logging

from gias2.registration import alignment_fitting as af
from gias2.registration import RBF
from gias2.mesh import vtktools

def register(source, target, init_rot, pts_only=False, out=None, view=False, **rbfregargs):
    
    if pts_only:
        source_points = source
        target_points = target
    else:
        source_points = source.v
        target_points = target.v

    #=============================================================#
    # rigidly register source points to target points
    init_trans = target_points.mean(0) - source_points.mean(0)
    t0 = np.hstack([init_trans, init_rot])
    reg1_T, source_points_reg1, reg1_errors = af.fitDataRigidDPEP(
                                                source_points,
                                                target_points,
                                                xtol=1e-6,
                                                sample=1000,
                                                t0=t0,
                                                outputErrors=1
                                                )

    # add isotropic scaling to rigid registration
    reg2_T, source_points_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
                                                source_points,
                                                target_points,
                                                xtol=1e-6,
                                                sample=1000,
                                                t0=np.hstack([reg1_T, 1.0]),
                                                outputErrors=1
                                                )

    #=============================================================#
    # rbf registration
    source_points_reg3, regRms, regRcf, regHist = RBF.rbfRegIterative(
        source_points_reg2, target_points, **rbfregargs
        )

    knots = regRcf.C

    #=============================================================#
    # create regstered mesh
    if pts_only:
        reg = source_points_reg3
    else:
        reg = copy.deepcopy(source)
        reg.v = source_points_reg3

    if out:
        if pts_only:
            n = np.arange(1,len(reg))
            _out = np.hstack([n[:,np.newaxis], reg])
            np.savetxt(
                args.out, _out, delimiter=',',
                fmt=['%6d', '%10.6f', '%10.6f', '%10.6f'],
                header='# rbf registered points'
                )
        else:
            writer = vtktools.Writer(v=reg.v, f=reg.f)
            writer.write(out)

    #=============================================================#
    # view
    if view:
        try:
            from gias2.visualisation import fieldvi
            has_mayavi = True
        except ImportError:
            has_mayavi = False

        if has_mayavi:
            v = fieldvi.Fieldvi()
            if pts_only:
                v.addData('target', target, renderArgs={'color':(1,0,0)})
                v.addData('source', source, renderArgs={'color':(0,1,0)})
                v.addData('source morphed', reg, renderArgs={'color':(0.3,0.3,1)})
            else:
                v.addTri('target', target, renderArgs={'color':(1,0,0)})
                v.addTri('source', source, renderArgs={'color':(0,1,0)})
                v.addTri('registered', reg, renderArgs={'color':(0.3,0.3,1)})
            
            v.addData('source points reg 2', source_points_reg2, renderArgs={'mode':'point'})
            v.addData('knots', knots, renderArgs={'mode':'sphere', 'color':(0,1.0,0), 'scale_factor':2.0})
            v.scene.background=(0,0,0)
            v.configure_traits()
        else:
            print('Visualisation error: cannot import mayavi')

    return reg, regRms, regRcf

def main_2_pass(args):
    print('{} to {}'.format(args.source,args.target))
    if args.points_only:
        source = np.loadtxt(args.source, skiprows=1, use_cols=(1,2,3))
    else:
        source = vtktools.loadpoly(args.source)
    
    if args.points_only:
        target = np.loadtxt(args.target, skiprows=1, use_cols=(1,2,3))
    else:
        target = vtktools.loadpoly(args.target)
    
    init_rot = np.deg2rad((0,0,0))

    rbfargs1 = {
        'basisType': 'gaussianNonUniformWidth',
        'basisArgs': {'s':1.0, 'scaling':1000.0},
        'distmode': 'alt',
        'xtol': 1e-1,
        'maxIt': 20,
        'maxKnots': 500,
        'minKnotDist': 10.0,
    }
    reg_1, rms1, rcf1 = register(source, target, init_rot, pts_only=args.points_only,
        out=False, view=False, **rbfargs1
        )

    rbfargs2 = {
        'basisType': 'gaussianNonUniformWidth',
        'basisArgs': {'s':1.0, 'scaling':10.0},
        'distmode': 'alt',
        'xtol': 1e-2,
        'maxIt': 20,
        'maxKnots': 1000,
        'minKnotDist': 2.5,
    }
    reg_2, rms2, rcf2 = register(reg_1, target, init_rot, pts_only=args.points_only,
        out=args.out, view=args.view, **rbfargs2
        )

    logging.info('{}, rms: {}'.format(path.split(args.target)[1], rms2))

def main_n_pass(args):
    print('RBF Registering {} to {}'.format(args.source,args.target))
    if args.points_only:
        source = np.loadtxt(args.source, skiprows=1, use_cols=(1,2,3))
    else:
        source = vtktools.loadpoly(args.source)
    
    if args.points_only:
        target = np.loadtxt(args.target, skiprows=1, use_cols=(1,2,3))
    else:
        target = vtktools.loadpoly(args.target)
    
    init_rot = np.deg2rad((0,0,0))

    # TODO
    rbfargs = load_rbf_config(args.rbfconfig)
    n_iterations = len(rbfargs)

    for it, rbfargs_i in enumerate(rbfargs):
        logging.info('Iteration {}'.format(it+1))
        if it==(n_iterations-1):
            reg_i, rms_i, rcf_i = register(source, target, init_rot, pts_only=args.points_only,
                out=args.out, view=args.view, **rbfargs_i
                )
        else:
            reg_i, rms_i, rcf_i = register(source, target, init_rot, pts_only=args.points_only,
                out=False, view=False, **rbfargs_i
                )

        source = reg_i

    logging.info('{}, rms: {}'.format(path.split(args.target)[1], rms_i))

def _load_rbf_config(fname):
    """
    Load the rbf registration config file
    """

    # TODO
    return

def main():
    parser = argparse.ArgumentParser(description='Non-rigid registration using a radial basis function.')
    parser.add_argument(
        '-s', '--source',
        help='file path of the source model.'
        )
    parser.add_argument(
        '-t', '--target',
        help='file path of the target model.'
        )
    parser.add_argument(
        '-o', '--out',
        help='file path of the output registered model.'
        )
    parser.add_argument(
        '-p', '--points-only',
        help='''Model are point clouds only. Expected file format is 1 header 
line, then n,x,y,z on each line after. UNTESTED'''
        )
    parser.add_argument(
        '-b', '--batch',
        help='file path of a list of model paths to fit. 1st model on list will be the source.'
        )
    parser.add_argument(
        '-d', '--outdir',
        help='directory path of the output registered models when using batch mode.'
        )
    parser.add_argument(
        '--outext',
        choices=('.obj', '.wrl', '.stl', '.ply', '.vtp'),
        help='output file extension. Ignored if --out is given, useful in batch mode.'
        )
    parser.add_argument(
        '-v', '--view',
        action='store_true',
        help='Visualise measurements and model in 3D'
        )
    parser.add_argument(
        '-l', '--log',
        help='log file'
        )
    args = parser.parse_args()

    # start logging
    if args.log:
        log_fmt = '%(levelname)s - %(asctime)s: %(message)s'
        log_level = logging.INFO

        logging.basicConfig(
            filename=args.log,
            level=log_level,
            format=log_fmt,
            )
        logging.info(
            'Starting RBF registration',
            )

    if args.batch is None:
        main_2_pass(args)
    else:
        model_paths = np.loadtxt(args.batch, dtype=str)
        args.source = model_paths[0]
        out_dir = args.outdir
        for i, mp in enumerate(model_paths):
            args.target = mp
            _p, _ext = path.splitext(path.split(mp)[1])
            if args.outext is not None:
                _ext = args.outext
            args.out = path.join(out_dir, _p+'_rbfreg'+_ext)
            main_2_pass(args)
            # main_n_pass(args)

if __name__=='__main__':
    main()