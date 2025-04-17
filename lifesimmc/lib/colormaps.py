import matplotlib.colors as mcolors

hex_colors = ['#000000', '#2749f4', '#4aaaf9', '#FFFFFF']
rgb_colors = [mcolors.hex2color(c) for c in hex_colors]
cmap_blue = mcolors.LinearSegmentedColormap.from_list("custom_cmap", rgb_colors, N=256)
