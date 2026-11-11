from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import xarray as xr
from dask.callbacks import Callback
import rasterio
from geosave.geodata.io.formats.geotiff import write_cog
from tests.geodata.conftest import build_raster

results={}
source=build_raster(times=2).chunk()
source=source.assign_coords(time_bounds=(('time','edge'),np.array([['2025-06-01','2025-06-01'],['2025-06-02','2025-06-02']],dtype='datetime64[ns]')))
tasks=[]
with Callback(pretask=lambda key,*args: tasks.append(key)):
    datasets=[]
    for time,dataset in source.groupby('time'):
        dataset=dataset.squeeze('time',drop=False)
        assert dataset.time.ndim==0
        assert dataset.time_bounds.dims==('edge',)
        assert dataset.gs.geobox==source.gs.geobox
        assert dataset.red.encoding==source.red.encoding
        datasets.append(dataset)
    assert not tasks
results['time_grouping']={'groups':len(datasets),'dask_tasks':len(tasks),'grid_encoding_bounds':'preserved'}
array=datasets[0].red
results['variable_extraction']={'dataset_has_time_bounds':'time_bounds' in datasets[0].coords,'array_has_time_bounds':'time_bounds' in array.coords}
coarse=source.assign_coords(x=source.x.values*2,y=source.y.values*2)
tree=xr.DataTree.from_dict({'fine':source,'coarse':coarse})
for path,node in tree.subtree_with_keys:
    if not node.data_vars:
        continue
    for time,dataset in node.to_dataset().groupby('time'):
        dataset=dataset.squeeze('time',drop=False)
        assert dataset.gs.geobox==node.to_dataset().gs.geobox
results['tree']={'different_grids_preserved':True,'same_times':True}
with TemporaryDirectory() as folder:
    try:
        write_cog(datasets[0],Path(folder)/'bounds.tif')
    except Exception as error:
        results['cog_with_bounds']={'error':type(error).__name__,'message':str(error)}
    path=write_cog(datasets[0].drop_vars('time_bounds'),Path(folder)/'scene.tif')
    with rasterio.open(path) as raster:
        assert raster.count==2
        np.testing.assert_array_equal(raster.read(1),datasets[0].red.values)
        assert raster.transform==datasets[0].gs.geobox.transform
    results['cog']={'bands':2,'raw_values_and_transform':'passed'}
duplicate=source.assign_coords(time=[source.time.values[0],source.time.values[0]])
for time,dataset in duplicate.groupby('time'):
    results['duplicate_time']={'group_size':dataset.sizes['time']}
    try:
        dataset.squeeze('time',drop=False)
    except ValueError:
        results['duplicate_time']['cannot_squeeze']=True
print(json.dumps(results,indent=2))
