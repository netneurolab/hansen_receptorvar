"""
receptor standard deviations
"""

import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.datasets import fetch_atlas_schaefer_2018
from neuromaps.parcellate import Parcellater
from netneurotools import datasets, plotting
from scipy.stats import zscore, spearmanr
import pyvista as pv
import matplotlib.cm as cm


def plot_subcortex(data, atlas_rois, hemi_labels, cmap="plasma",
                   vmin=None, vmax=None, outpath=None):

    assert len(data) == len(atlas_rois), \
        "Data length must match number of ROIs"

    views = {
        "Right Lateral": ("zy", (0, 0, -1)),
        "Left Lateral": ("yz", (0, 0, -1)),
        "Right Medial": ("yz", (0, 0, -1)),
        "Left Medial": ("zy", (0, 0, -1)),
    }

    plotters = []
    for view_name, (cam_pos, up_dir) in views.items():
        hemi = 'R' if 'Right' in view_name else 'L'
        pl = pv.Plotter(off_screen=True, window_size=(512, 512))
        for i, (roi, hem) in enumerate(zip(atlas_rois, hemi_labels)):
            if hem != hemi:
                continue
            color = cm.get_cmap(cmap)((data[i] - (vmin or np.min(data))) /
                                      ((vmax or np.max(data))
                                       - (vmin or np.min(data))))
            if np.isnan(data[i]):
                color = [0.7, 0.7, 0.7]
            pl.add_mesh(roi, color=color[:3])
        pl.camera_position = cam_pos
        pl.camera.up = up_dir
        plotters.append(pl)

    # Render all views and combine into 2x2 panel
    imgs = [pl.screenshot(return_img=True) for pl in plotters]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    for ax, img, title in zip(axs, imgs, views.keys()):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath)


"""
set up
"""

path = "/home/jhansen/gitrepos/hansen_receptor-variability/"

filenames = [path + 'data/PET_volumes/5HT1a_cumi101_hc8_beliveau',
             path + 'data/PET_volumes/5HT1b_az_hc36_beliveau',
             path + 'data/PET_volumes/5HT2a_cimbi_hc29_beliveau',
             path + 'data/PET_volumes/5HT4_sb20_hc59_beliveau',
             path + 'data/PET_volumes/5HTT_dasb_hc100_beliveau',
             path + 'data/PET_volumes/5HTT_madam_hc49_nummenmaa',
             path + 'data/PET_volumes/CB1_fmpepVt2_hc20_nummenmaa',
             path + 'data/PET_volumes/D2_raclopride_hc16_tuominen',
             path + 'data/PET_volumes/D2_raclopride_hc47_nummenmaa',
             path + 'data/PET_volumes/D2_fallypride_hc49_jaworska',
             path + 'data/PET_volumes/DOPA_fdopa_hc17_nummenmaa',
             path + 'data/PET_volumes/GABAa1_ro154513_hc23_chang',
             path + 'data/PET_volumes/GABAa5_ro154513_hc23_chang',
             path + 'data/PET_volumes/GABAabz_flumazenil_hc16_norgaard',
             path + 'data/PET_volumes/MOR_carfentanil_hc86_nummenmaa',
             path + 'data/PET_volumes/mGluR5_ABP688_hc27_dubois',
             path + 'data/PET_volumes/mGluR5_ABP688_hc73_smart',
             path + 'data/PET_volumes/NMDA_ge179_hc29_galovic',
             path + 'data/PET_volumes/SV2A_ucbj_hc32_nummenmaa',
             path + 'data/PET_volumes/VAChT_feobv_hc25_tuominen']

# load if already saved:
data = pd.read_pickle(path + 'data/mean_var_dataframes.pkl')
(receptor_means, receptor_sds, cv,
 receptor_means_tian, receptor_sds_tian, cv_tian) = (
    data['mean_schaefer'], data['sd_schaefer'], data['cv_schaefer'],
    data['mean_tian'], data['sd_tian'], data['cv_tian']
)

"""
fetch and parcellate mean and sd images
"""

# Schaefer-100 (cortex)
schaefer = fetch_atlas_schaefer_2018(n_rois=100)

parcellated_mean = {}
parcellated_sd = {}
cv = {}
parcellater = Parcellater(schaefer['maps'], 'MNI152')

for receptor in filenames:
    name = receptor.split('/')[-1].split('.')[0]

    # Parcellate mean and std images
    mean_vals = parcellater.fit_transform(receptor + '_mean.nii.gz',
                                          'MNI152', True).squeeze()
    std_vals = parcellater.fit_transform(receptor + '_sd.nii.gz',
                                         'MNI152', True).squeeze()

    # Step 1: set values < 0 to 0
    mean_vals[mean_vals < 0] = 0

    # Step 2: if any values are < 0.1, set the bottom 5% to 0
    if np.any(mean_vals < 0.1):
        threshold = np.percentile(mean_vals, 5)
        mean_vals[mean_vals < threshold] = 0

    # Save cleaned results
    parcellated_mean[name] = mean_vals
    parcellated_sd[name] = std_vals

    # Compute CV: std / mean, but set CV to nan if mean == 0
    cv_vals = np.full_like(mean_vals, np.nan)
    nonzero_mask = mean_vals != 0
    cv_vals[nonzero_mask] = std_vals[nonzero_mask] / mean_vals[nonzero_mask]
    cv[name] = cv_vals

# Tian S4 (subcortex)
parcellater_tian = Parcellater(path + 'data/Tian_Subcortex_S4_3T_1mm.nii.gz',
                               'MNI152')
with open(path+'data/Tian_Subcortex_S4_3T_label.txt', 'r') as f:
    tianlabels = [line.strip() for line in f]

parcellated_mean_tian = {}
parcellated_sd_tian = {}
cv_tian = {}

for receptor in filenames:
    name = receptor.split('/')[-1].split('.')[0]

    # Parcellate mean and std images
    mean_vals = parcellater_tian.fit_transform(receptor + '_mean.nii.gz',
                                               'MNI152', True).squeeze()
    std_vals = parcellater_tian.fit_transform(receptor + '_sd.nii.gz',
                                              'MNI152', True).squeeze()

    # Step 1: set values < 0 to 0
    mean_vals[mean_vals < 0] = 0

    # Step 2: if any values are < 0.1, set the bottom 5% to 0
    if np.any(mean_vals < 0.1):
        threshold = np.percentile(mean_vals, 5)
        mean_vals[mean_vals < threshold] = 0

    # Save cleaned results
    parcellated_mean_tian[name] = mean_vals
    parcellated_sd_tian[name] = std_vals

    # Compute CV: std / mean, but set CV to nan if mean == 0
    cv_vals = np.full_like(mean_vals, np.nan)
    nonzero_mask = mean_vals != 0
    cv_vals[nonzero_mask] = std_vals[nonzero_mask] / mean_vals[nonzero_mask]
    cv_tian[name] = cv_vals

# make into dataframe
receptor_means = pd.DataFrame(parcellated_mean, index=schaefer['labels'])
receptor_sds = pd.DataFrame(parcellated_sd, index=schaefer['labels'])
cv = pd.DataFrame(cv, index=schaefer['labels'])

receptor_means_tian = pd.DataFrame(parcellated_mean_tian, index=tianlabels)
receptor_sds_tian = pd.DataFrame(parcellated_sd_tian, index=tianlabels)
cv_tian = pd.DataFrame(cv_tian,
                       index=tianlabels)

# save out
data = {
    'mean_schaefer': receptor_means,
    'sd_schaefer': receptor_sds,
    'cv_schaefer': cv,
    'mean_tian': receptor_means_tian,
    'sd_tian': receptor_sds_tian,
    'cv_tian': cv_tian
}
pd.to_pickle(data, path+'data/mean_var_dataframes.pkl')

"""
plot cortex
"""

M = receptor_means.copy()
S = receptor_sds.copy()
C = cv.copy()

annot = datasets.fetch_schaefer2018('fsaverage')['100Parcels7Networks']
occipital_lobe = np.where([
    'Vis' in s for s in schaefer['labels'].astype('str')])[0]

# plot each receptor MEAN map
for key, value in M.items():
    if key == 'DOPA_fdopa_hc17_nummenmaa' or \
       key == 'MOR_carfentanil_hc86_nummenmaa':
        value[occipital_lobe] = np.nan

    brain = plotting.plot_fsaverage(data=value,
                                    lhannot=annot.lh,
                                    rhannot=annot.rh,
                                    colormap='plasma',
                                    views=['lat', 'med'],
                                    data_kws={'representation': "wireframe",
                                              'line_width': 4.0})
    brain.save_image(path + 'figures/png/schaefer100/surface_receptor_'
                     + key + '_mean.png')

# plot each receptor STANDARD DEVIATION map
for key, value in S.items():
    if key == 'DOPA_fdopa_hc17_nummenmaa' or\
       key == 'MOR_carfentanil_hc86_nummenmaa':
        value[occipital_lobe] = np.nan

    brain = plotting.plot_fsaverage(data=value,
                                    lhannot=annot.lh,
                                    rhannot=annot.rh,
                                    colormap='plasma',
                                    views=['lat', 'med'],
                                    data_kws={'representation': "wireframe",
                                              'line_width': 4.0})
    brain.save_image(path+'figures/png/schaefer100/surface_receptor_'
                     + key + '_st.png')

# plot each receptor COEFFICIENT OF VARIATION map
for key, value in C.items():
    if key == 'DOPA_fdopa_hc17_nummenmaa' or\
       key == 'MOR_carfentanil_hc86_nummenmaa':
        value[occipital_lobe] = np.nan

    brain = plotting.plot_fsaverage(data=value,
                                    lhannot=annot.lh,
                                    rhannot=annot.rh,
                                    colormap='plasma',
                                    vmin=value.min(),
                                    vmax=value.max(),
                                    views=['lat', 'med'],
                                    data_kws={'representation': "wireframe",
                                              'line_width': 4.0})
    brain.save_image(path+'figures/png/schaefer100/surface_receptor_'
                     + key + '_cv.png')

"""
plot subcortex
"""

M = receptor_means_tian.copy()
S = receptor_sds_tian.copy()
C = cv_tian.copy()

atlas = pv.read(path+'data/Tian_Subcortex_S4_3T_1mm.nii.gz')
atlas_rois = [atlas.image_threshold([i, i]).contour(
    [1]).smooth_taubin(n_iter=25, pass_band=0.01, non_manifold_smoothing=True)
              for i in range(1, 55)]
hemi_labels = ['R' if i < 28 else 'L' for i in range(len(atlas_rois))]

for key, value in M.items():
    plot_subcortex(
        data=value,
        atlas_rois=atlas_rois,
        cmap='plasma',
        hemi_labels=hemi_labels,
        vmin=value.min(),
        vmax=value.max(),
        outpath=path+'figures/png/tianS4/surface_' + key + '_means.png'
        )

for key, value in S.items():
    plot_subcortex(
        data=value,
        atlas_rois=atlas_rois,
        cmap='plasma',
        hemi_labels=hemi_labels,
        vmin=value.min(),
        vmax=value.max(),
        outpath=path+'figures/png/tianS4/surface_' + key + '_sds.png'
        )

for key, value in C.items():
    plot_subcortex(
        data=value,
        atlas_rois=atlas_rois,
        cmap='plasma',
        hemi_labels=hemi_labels,
        vmin=value.min(),
        vmax=value.max(),
        outpath=path+'figures/png/tianS4/surface_' + key + '_cv.png'
        )

"""
correlate means and stds
"""

no_occipital = np.where([
    'Vis' not in s for s in schaefer['labels'].astype('str')])[0]
striatum = [i for i, label in enumerate(tianlabels)
            if any(sub in label for sub in ['NAc', 'CAU', 'PUT'])]

fig, axs = plt.subplots(4, 5, figsize=(25, 13))
axs = axs.ravel()

for i, key in enumerate(M.keys()):

    if M.shape[0] == 100:  # schaefer-100
        if key == 'DOPA_fdopa_hc17_nummenmaa' or\
           key == 'MOR_carfentanil_hc86_nummenmaa':
            axs[i].scatter(M[key][no_occipital], S[key][no_occipital],
                           c=C[key][no_occipital], cmap='plasma')
            r, p = spearmanr(M[key][no_occipital], S[key][no_occipital])
        else:
            axs[i].scatter(M[key], S[key], c=C[key], cmap='plasma')
            r, p = spearmanr(M[key], S[key])

    elif M.shape[0] == 54:  # tian-S4
        if key == 'DOPA_fdopa_hc17_nummenmaa' \
            or key == 'D2_raclopride_hc16_tuominen' \
                or key == 'D2_raclopride_hc47_nummenmaa':
            striatum_edge = np.zeros((len(M[key]), ))
            striatum_edge[striatum] = 1
            axs[i].scatter(M[key], S[key], c=C[key],
                           linewidth=striatum_edge, edgecolor='k',
                           cmap='plasma')
            r, p = spearmanr(M[key][striatum], S[key][striatum])
        else:
            axs[i].scatter(M[key], S[key], c=C[key], cmap='plasma')
            r, p = spearmanr(M[key], S[key])

    axs[i].set_title(key)
    axs[i].set_xlabel('mean')
    axs[i].set_ylabel('sd')
    axs[i].set_aspect(1.0/axs[i].get_data_ratio(), adjustable='box')
fig.tight_layout()
fig.savefig(path+'figures/eps/scatter_mean_sd_tianS4.eps')

"""
histograms of cv distribution
"""

fig, axs = plt.subplots(len(M.keys()), 1, figsize=(10, 20), sharex=True)
axs = axs.ravel()

for i, key in enumerate(M.keys()):
    if M.shape[0] == 100:  # schaefer-100
        if key == 'DOPA_fdopa_hc17_nummenmaa' or\
           key == 'MOR_carfentanil_hc86_nummenmaa':
            sns.histplot(C[key][no_occipital], ax=axs[i], kde=True,
                         color='blue', label='No Occipital')
            axs[i].axvline(np.std(M[key][no_occipital]) /
                           np.mean(M[key][no_occipital]), color='red')
        else:
            sns.histplot(C[key], ax=axs[i], kde=True, color='gray')
            axs[i].axvline(np.std(M[key]) / np.mean(M[key]), color='red')

    elif M.shape[0] == 54:  # tian-S4
        if key in ['DOPA_fdopa_hc17_nummenmaa', 'D2_raclopride_hc16_tuominen',
                   'D2_raclopride_hc47_nummenmaa']:
            sns.histplot(C[key], ax=axs[i], kde=True,
                         color='gray', label='All')
            sns.histplot(C[key][striatum], ax=axs[i], kde=True,
                         color='green', label='Striatum')
            axs[i].axvline(np.std(M[key][striatum]) /
                           np.mean(M[key][striatum]),
                           color='green', label='Striatum')
            axs[i].axvline(np.std(M[key]) / np.mean(M[key]),
                           color='red', label='All')

        else:
            sns.histplot(C[key], ax=axs[i], kde=True, color='gray')
            axs[i].axvline(np.std(M[key]) / np.mean(M[key]), color='red')

    axs[i].set_title(key)

fig.tight_layout()
fig.savefig(path + 'figures/eps/hist_cv_tianS4.eps')

"""
compare across regions vs across people variability
"""

# variance
values = {key: np.var(M[key]) / np.mean(S[key] ** 2) for key in M.keys()}

# cv
cvratio = {key: (np.std(M[key]) / np.mean(M[key])) / np.mean(C[key])
           for key in M.keys()}
sorted_items = sorted(values.items(), key=lambda x: x[1])

# now compare this with replicability using other tracers
recpath = '/home/jhansen/gitrepos/hansen_receptors/data/'\
    + 'PET_parcellated/scale100/'
recpath = '/home/jhansen/projects/proj_receptors/PET_parcellated/tianS4/'

receptors_csv = [recpath + '5HT1a_way_hc36_savli.csv',
                 recpath + '5HT1b_p943_hc22_savli.csv',
                 recpath + '5HT1b_p943_hc65_gallezot.csv',
                 recpath + '5HT2a_alt_hc19_savli.csv',
                 recpath + '5HT2a_mdl_hc3_talbot.csv',
                 recpath + '5HTT_dasb_hc30_savli.csv',
                 recpath + 'CB1_FMPEPd2_hc22_laurikainen.csv',
                 recpath + 'CB1_omar_hc77_normandin.csv',
                 recpath + 'D2_flb457_hc37_smith.csv',
                 recpath + 'D2_flb457_hc55_sandiego.csv',
                 recpath + 'D2_raclopride_hc7_alakurtti.csv',
                 recpath + 'GABAa_flumazenil_hc6_dukart.csv',
                 recpath + 'mGluR5_abp_hc22_rosaneto.csv',
                 recpath + 'MU_carfentanil_hc204_kantonen.csv',
                 recpath + 'MU_carfentanil_hc39_turtonen.csv',
                 recpath + 'VAChT_feobv_hc5_bedard_sum.csv',
                 recpath + 'VAChT_feobv_hc18_aghourian_sum.csv']

receptors_rep = {}
for r in receptors_csv:
    receptors_rep[r.split('/')[-1].split('.')[0]] = zscore(
        np.genfromtxt(r, delimiter=','))
# change name to MOR
receptors_rep['MOR_carfentanil_hc204_kantonen'] = receptors_rep.pop(
    'MU_carfentanil_hc204_kantonen')
receptors_rep['MOR_carfentanil_hc39_turtonen'] = receptors_rep.pop(
    'MU_carfentanil_hc39_turtonen')

rep_corr = {}

for col in M.columns:

    # don't consider raclopride if we're in the cortex
    # if M.shape[0] == 100 and 'D2_raclopride' in col:
    #     continue

    x = M[col].squeeze()  # receptor map

    rep_corr[col] = []
    base_key = col.split('_')[0]  # receptor name

    # handle GABAa-bz maps 
    if base_key[:5] == 'GABAa':
        base_key = 'GABAa'

    # first correlate with repeats in M
    if base_key == '5HTT' or base_key == 'D2' or base_key == 'GABAa' or base_key == 'mGluR5':
        if base_key == 'GABAa':
            addedmap = [k for k in M if k.split('_')[0][:5] == base_key
                        and k != col]
        elif base_key == 'D2' and M.shape[0] == 100 and 'raclopride' not in col:
            # only compare with non-raclopride D2 maps
            addedmap = [k for k in M if k.split('_')[0] == base_key
                        and 'raclopride' not in k and k != col]
        else:
            addedmap = [k for k in M if k.split('_')[0] == base_key and k != col]

        for addedmap in addedmap:
            r, _ = spearmanr(x, M[addedmap].squeeze(), nan_policy='omit')
            rep_corr[col].append(r)

    if M.shape[0] == 100 and base_key == 'D2' and 'raclopride' not in col:
        matched_keys = [k for k in receptors_rep if k.split('_')[0] == base_key
                        and 'raclopride' not in k]
    else:
        matched_keys = [k for k in receptors_rep if k.split('_')[0] == base_key]

    if not matched_keys:
        continue  # Skip if no matches in receptors_rep

    for k in matched_keys:
        y = receptors_rep[k].squeeze()
        r, _ = spearmanr(x, y, nan_policy='omit')
        rep_corr[col].append(r)

fig, ax = plt.subplots()
x = []
y = []
for key in rep_corr.keys():
    mean = np.mean(rep_corr[key])
    if np.isnan(mean):
        continue
    x.append(mean)
    y.append(cvratio[key])
    ax.scatter(mean, cvratio[key], c='blue')
    ax.text(mean, cvratio[key], key.split('_')[0])
r, p = spearmanr(x, y)
ax.set_title(f'r={r:.2f}, p={p:.3f}')
ax.set_xlabel('mean replicability')
ax.set_ylabel('across region cv / across subject cv')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
fig.savefig(path+'figures/eps/scatter_replicability_ctx.eps')
