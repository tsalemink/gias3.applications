#!/usr/bin/env python
"""
FILE: giasrbfreg.py
LAST MODIFIED: 19/03/18
DESCRIPTION:
Script for registering one model to another using RBFs.

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""

import argparse
import configparser
import copy
import logging
from os import path

import numpy as np
import sys

from gias3.applications.general import init_log
from gias3.mesh import vtktools
from gias3.registration import RBF
from gias3.registration import alignment_fitting as af

log = logging.getLogger(__name__)


def register(source, target, init_rot, pts_only=False, out=None, view=False, **rbfregargs):
    if pts_only:
        source_points = source
        target_points = target
    else:
        source_points = source.v
        target_points = target.v

    # =============================================================#
    # rigidly register source points to target points
    init_trans = target_points.mean(0) - source_points.mean(0)
    t0 = np.hstack([init_trans, init_rot])
    reg1_T, source_points_reg1, reg1_errors = af.fitDataRigidDPEP(
        source_points,
        target_points,
        xtol=1e-6,
        sample=1000,
        t0=t0,
        output_errors=1
    )

    # add isotropic scaling to rigid registration
    reg2_T, source_points_reg2, reg2_errors = af.fitDataRigidScaleDPEP(
        source_points,
        target_points,
        xtol=1e-6,
        sample=1000,
        t0=np.hstack([reg1_T, 1.0]),
        output_errors=1
    )

    # =============================================================#
    # rbf registration
    source_points_reg3, regRms, regRcf, regHist = RBF.rbfRegIterative(
        source_points_reg2, target_points, **rbfregargs
    )
    # source_points_reg3, regRms, regRcf, regHist = RBF.rbfRegIterative2(
    #     source_points_reg2, target_points, **rbfregargs
    #     )
    # source_points_reg3, regRms, regRcf, regHist = RBF.rbfRegIterative3(
    #     source_points_reg2, target_points, **rbfregargs
    #     )

    knots = regRcf.C

    # =============================================================#
    # create regstered mesh
    if pts_only:
        reg = source_points_reg3
    else:
        reg = copy.deepcopy(source)
        reg.v = source_points_reg3

    if out:
        if pts_only:
            n = np.arange(1, len(reg) + 1)
            _out = np.hstack([n[:, np.newaxis], reg])
            np.savetxt(
                out, _out, delimiter=', ',
                fmt=['%8d', '%10.6f', '%10.6f', '%10.6f'],
                header='rbf registered points'
            )
        else:
            writer = vtktools.Writer(v=reg.v, f=reg.f)
            writer.write(out)

    # =============================================================#
    # view
    if view:
        try:
            from gias3.visualisation import fieldvi
            has_mayavi = True
        except ImportError:
            has_mayavi = False

        if has_mayavi:
            v = fieldvi.FieldVi()
            if pts_only:
                v.addData('target', target, render_args={'color': (1, 0, 0), 'mode': 'point'})
                v.addData('source', source, render_args={'color': (0, 1, 0), 'mode': 'point'})
                v.addData('source morphed', reg, render_args={'color': (0.3, 0.3, 1), 'mode': 'point'})
            else:
                v.addTri('target', target, render_args={'color': (1, 0, 0)})
                v.addTri('source', source, render_args={'color': (0, 1, 0)})
                v.addTri('registered', reg, render_args={'color': (0.3, 0.3, 1)})

            v.addData('source points reg 2', source_points_reg2, render_args={'mode': 'point'})
            v.addData('knots', knots, render_args={'mode': 'sphere', 'color': (0, 1.0, 0), 'scale_factor': 2.0})
            v.scene.background = (0, 0, 0)
            v.start()

        else:
            log.info('Visualisation error: cannot import mayavi')

    return reg, regRms, regRcf


# def register_2_pass(args):
#     log.info('RBF Registering {} to {}'.format(args.source,args.target))
#     if args.points_only:
#         source = np.loadtxt(args.source, skiprows=1, use_cols=(1,2,3))
#     else:
#         source = vtktools.loadpoly(args.source)

#     if args.points_only:
#         target = np.loadtxt(args.target, skiprows=1, use_cols=(1,2,3))
#     else:
#         target = vtktools.loadpoly(args.target)

#     init_rot = np.deg2rad((0,0,0))

#     rbfargs1 = {
#         'basisType': 'gaussianNonUniformWidth',
#         'basisArgs': {'s':1.0, 'scaling':1000.0},
#         'distmode': 'alt',
#         'xtol': 1e-1,
#         'maxIt': 20,
#         'maxKnots': 500,
#         'minKnotDist': 20.0,
#         'maxKnotsPerIt': 20,
#     }
#     reg_1, rms1, rcf1 = register(source, target, init_rot, pts_only=args.points_only,
#         out=False, view=False, **rbfargs1
#         )

#     rbfargs2 = {
#         'basisType': 'gaussianNonUniformWidth',
#         'basisArgs': {'s':1.0, 'scaling':10.0},
#         'distmode': 'alt',
#         'xtol': 1e-3,
#         'maxIt': 20,
#         'maxKnots': 1000,
#         'minKnotDist': 2.5,
#         'maxKnotsPerIt': 20,
#     }
#     reg_2, rms2, rcf2 = register(reg_1, target, init_rot, pts_only=args.points_only,
#         out=args.out, view=args.view, **rbfargs2
#         )

#     logging.info('{}, rms: {}'.format(path.split(args.target)[1], rms2))

#     return source, target, (reg_1, rms1, rcf1), (reg_2, rms2, rcf2)

def register_n_pass(args):
    log.info('RBF Registering {} to {}'.format(args.source, args.target))
    if args.points_only:
        source = np.loadtxt(args.source, skiprows=1, usecols=(1, 2, 3), delimiter=',')
    else:
        source = vtktools.loadpoly(args.source)

    if args.points_only:
        target = np.loadtxt(args.target, skiprows=1, usecols=(1, 2, 3), delimiter=',')
    else:
        target = vtktools.loadpoly(args.target)

    init_rot = np.deg2rad((0, 0, 0))

    rbfargs = parse_config(args.config)
    n_iterations = len(rbfargs)
    _source = source
    for it, rbfargs_i in enumerate(rbfargs):
        logging.info('Registration pass {}'.format(it + 1))
        if it != (n_iterations - 1):
            reg_i, rms_i, rcf_i = register(source, target, init_rot, pts_only=args.points_only,
                                           out=False, view=False, **rbfargs_i
                                           )
        else:
            # last iteration
            reg_i, rms_i, rcf_i = register(source, target, init_rot, pts_only=args.points_only,
                                           out=args.out, view=args.view, **rbfargs_i
                                           )

        source = reg_i

    logging.info('{}, rms: {}'.format(path.split(args.target)[1], rms_i))

    return source, target, (reg_i, rms_i, rcf_i)


# DEFAULT 2 pass parameters
DEFAULT_PARAMS = [
    {
        'basis_type': 'gaussianNonUniformWidth',
        'basis_args': {'s': 1.0, 'scaling': 1000.0},
        'dist_mode': 'alt',
        'xtol': 1e-1,
        'max_it': 20,
        'max_knots': 500,
        'min_knot_dist': 20.0,
        'max_knots_per_it': 20,
    },
    {
        'basis_type': 'gaussianNonUniformWidth',
        'basis_args': {'s': 1.0, 'scaling': 10.0},
        'dist_mode': 'alt',
        'xtol': 1e-3,
        'max_it': 20,
        'max_knots': 1000,
        'min_knot_dist': 2.5,
        'max_knots_per_it': 20,
    }
]


def parse_config(fname):
    if fname is None:
        return DEFAULT_PARAMS

    cfg = configparser.ConfigParser()
    cfg.read(fname)

    n_passes = cfg.getint('main', 'n_passes')

    params = []
    for _pass in range(1, n_passes + 1):
        sec = 'pass_{:d}'.format(_pass)
        pass_params = {
            'basisType': cfg.get(sec, 'basis_type'),
            'basisArgs': {
                's': 1.0,
                'scaling': cfg.getfloat(sec, 'basis_scaling')
            },
            'distmode': cfg.get(sec, 'dist_mode'),
            'xtol': cfg.getfloat(sec, 'xtol'),
            'maxIt': cfg.getint(sec, 'max_it'),
            'maxKnots': cfg.getint(sec, 'max_knots'),
            'minKnotDist': cfg.getfloat(sec, 'min_knot_dist'),
            'maxKnotsPerIt': cfg.getint(sec, 'max_knots_per_it'),
        }
        params.append(pass_params)

    return params


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
        '-c', '--config',
        default=None,
        help='file path of a configuration file for rbf registration pass parameters. See examples/rbfreg-params.ini for an example config file.'
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
    init_log(args.log)
    log.info('Starting RBF registration')

    if args.batch is None:
        register_n_pass(args)
    else:
        model_paths = np.loadtxt(args.batch, dtype=str)
        args.source = model_paths[0]
        out_dir = args.outdir
        for i, mp in enumerate(model_paths):
            args.target = mp
            _p, _ext = path.splitext(path.split(mp)[1])
            if args.outext is not None:
                _ext = args.outext
            args.out = path.join(out_dir, _p + '_rbfreg' + _ext)
            register_n_pass(args)


if __name__ == '__main__':
    main()
