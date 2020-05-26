'''
A module to aid common plot styling
'''
try:
    basestring
except:
    basestring = str

textwidth = 8.27 * 0.72 # JINST text width in inches

from enum import Enum
class figsize(Enum):
  FULLSQUARE = (textwidth, textwidth)
  HALFSQUARE = (textwidth/2, textwidth/2)
  FULLWIDTH_HALFHEIGHT = (textwidth, textwidth/2)

def init(nLines=5):
  import matplotlib as mpl
  from cycler import cycler
  mpl.rcParams['font.size'] = 8
  #mpl.rcParams['text.usetex'] = True
  mpl.rcParams['font.family'] = 'serif'
  mpl.rcParams['font.sans-serif'] = 'Computer Modern Sans serif'
  mpl.rcParams['figure.figsize'] = (textwidth, textwidth)
  mpl.rcParams['legend.fontsize'] = 7
  mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{siunitx}']
  if nLines is not None:
    if nLines <= 5:
      cols = colors#list(reversed(colors_sequential_5))
    else:
      cols = list(reversed(colors_sequential_9))
    mpl.rcParams['axes.prop_cycle'] = cycler(color=cols)

colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
colors_sequential_5 = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
colors_sequential_9 = ['#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
#colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

def finalisefig(adjust=False):
  import matplotlib.pyplot as plt
  if adjust:
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
  plt.figtext(0.15, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='left', fontsize=14)

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    import matplotlib.colors as mcolors
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, basestring): low = c(low)
    if isinstance(high, basestring): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])
import matplotlib.colors as mcolors
rgb = mcolors.ColorConverter().to_rgb
colmap = diverge_map(rgb(colors[1]), rgb(colors[2]))

def add_logo(ax, ax_pos=(0.03, 0.77), width=0.35, zoom=0.06):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.patches import Rectangle
    # plot the hls4ml logo on the axis ax at position ax_pos (in axis units, i.e. 0 to 1)
    # with width width (also in axis units)
    # import the logo
    im = plt.imread('logo_crop.jpg')
    im_w = im.shape[1]
    im_h = im.shape[0]
    aspect = im_h / im_w
    height = aspect * width * 1.2
    width_pix = ax.transAxes.transform([width, width])[0]
    oim = OffsetImage(im, zoom=zoom, interpolation='bilinear')
    # Add one rectangle with white fill and 0.8 alpha, then a second rectangle
    # with no fill, to cover the edges
    rectangle = Rectangle(ax_pos, width, height, transform=plt.gca().transAxes, edgecolor='0.8', joinstyle='round', facecolor='white', alpha=0.8, clip_on=False)
    ax.add_patch(rectangle)
    rectangle = Rectangle(ax_pos, width, height, transform=plt.gca().transAxes, edgecolor='0.8', joinstyle='round', fill=False, clip_on=False)
    ax.add_patch(rectangle)
    box_xy = rectangle.get_patch_transform().transform([0.5, 0.5]) # rectangle local to axis
    box_xy = ax.transAxes.transform(box_xy) # axis to display units
    box_xy = ax.transData.inverted().transform(box_xy) # display units to data
    box = AnnotationBbox(oim, box_xy, frameon=False, box_alignment=(0.5, 0.5), pad=0., annotation_clip=False)
    ax.add_artist(box)

