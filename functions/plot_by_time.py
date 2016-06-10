# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.cm as cm
def plot_by_time(data, y, fmt = '-', datetime_to = None, 
                 datetime_from = None, granular = False,
                 color = None):
    plt.style.use('seaborn-muted')
    ind = ((data.datetime < datetime_to) | (datetime_to is None)) & \
          ((data.datetime >= datetime_from) | (datetime_from is None))
    data_used = data[ind]
    if color is None:
        plt.plot_date(x = 'datetime', y = y, fmt = fmt, data = data_used)
    else:
        x = mdates.date2num(data_used.datetime.astype(dt.datetime))
        plt.scatter(x = x, y = y, c = data_used[color].tolist(), 
                    data = data_used, cmap = cm.jet)
        plt.colorbar()
        axes = plt.gca()
        loc = mdates.AutoDateLocator()
        axes.xaxis.set_major_locator(loc)
        axes.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
        axes.set_xlim([x.min()-1,x.max()+1])
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    axes.set_ylim([ymin-(ymax-ymin)*0.05, ymax*1.1])
    if granular:
        xax = plt.gca().xaxis
        xax.set_major_locator(mdates.DayLocator())
        xax.set_major_formatter(mdates.DateFormatter('%d-%a'))
        plt.gcf().autofmt_xdate()
    

