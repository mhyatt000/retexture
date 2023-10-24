
import site
import os
pkg = site.getsitepackages()[0]
print(pkg)
os.system(f'export SITE={pkg}')
_ = '~/.anaconda3/envs/retexture/lib/python3.11/site-packages'
