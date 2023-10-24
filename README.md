# retexture
using blender to augment 3d model textures

<table>
  <tr>
    <td><img src="assets/elephant1_peacockwing2_0.0.png" alt="" width="400"/></td>
    <td><img src="assets/elephant1_Dunstickin-1154251706_0_0.0.png" alt="" width="400"/></td>
  </tr>
  <tr>
    <td><img src="assets/elephant1_Carpet_Berber_Multi_0.0.png" alt="" width="400"/></td>
    <td><img src="assets/elephant1_tetrabod_0.0.png" alt="" width="400"/></td>
  </tr>
</table>

## Installation

1. clone the repo (more instructions later)
2. `pip install -r requirements.txt`

### Installing blender

* MacOS: `brew install blender`
* Unix: `apt-get install blender`

### Install via script

see `./install.sh`

## Dataset Setup

place all models in `datasets/models` and place textures in `datasets/textures` like so:

```
datasets
├── models
│   ├── bird1.dae
│   ├── bird_duck.dae
│   ├── bird_duck2.dae
│   ├── bird_eagle.dae
│   ├── bird_raven.dae
│   ├── butterfly.dae
│   ├── butterfly1.dae
│   ...
├── datasets/textures
│   ├── _fish_bass
│   │   ├── Sketchy_Lines_Wavy_45_A.jpg
│   │   ├── __Chalk_1.jpg
│   │   ├── __Sketchy_Scales_1.jpg
│   │   ├── __Wavy_Lines_45deg_A_1.jpg
│   │   ├── material_1.jpg
│   │   ├── material_2.jpg
│   │   ├── material_3.jpg
│   │   └── material_5.jpg
│   ├── bird_duck
│   │   └── mallard_male1.jpg
│   ├── bird_duck2
│   │   └── Color_000.JPG
│   ...
```

## Run

`./run.sh`

TODO:
* `./run.sh <config>`
