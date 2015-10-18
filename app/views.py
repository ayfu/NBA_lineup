import sqlite3
import os
from contextlib import closing
from app import app
import StringIO
import Tkinter as tk
from collections import OrderedDict

from flask import Flask, request, session, g, redirect, url_for, \
                  abort, render_template, flash, make_response
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import numpy as np
from bokeh import mpl
from bokeh.embed import file_html, components
from bokeh.plotting import figure, ColumnDataSource, output_file, show
from bokeh.resources import INLINE
from bokeh.templates import JS_RESOURCES, CSS_RESOURCES
from bokeh.models import HoverTool
from bokeh.util.string import encode_utf8



#DATABASE = '../sql/nba_stats.db'
DATABASE = 'app/results.db'

app.config.from_object(__name__)

"""
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = connect_db()
    return db
"""
def connect_db():
    return sqlite3.connect(DATABASE)

@app.before_request
def before_request():
    g.db = connect_db()

@app.teardown_request
def teardown_request(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()

"""
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
"""
@app.route('/')
def index():
   user = '' # fake user
   return render_template("index.html",
       title = 'Home',
       user = user)

@app.route('/about')
def about():
   return render_template("about.html")

@app.route('/model')
def model():
   return render_template("model.html")



@app.route('/output')
def output():
    ALL = request.args.get('ALL')
    if ALL == "ALL":
        input_team = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL',
                      'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL',
                      'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC',
                      'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR',
                      'UTA', 'WAS']
    else:
        ATL = request.args.get('ATL')
        BOS = request.args.get('BOS')
        BKN = request.args.get('BKN')
        CHA = request.args.get('CHA')
        CHI = request.args.get('CHI')
        CLE = request.args.get('CLE')
        DAL = request.args.get('DAL')
        DET = request.args.get('DET')
        GSW = request.args.get('GSW')
        HOU = request.args.get('HOU')
        IND = request.args.get('IND')
        LAC = request.args.get('LAC')
        LAL = request.args.get('LAL')
        MEM = request.args.get('MEM')
        MIA = request.args.get('MIA')
        MIL = request.args.get('MIL')
        NOP = request.args.get('NOP')
        OKC = request.args.get('OKC')
        ORL = request.args.get('ORL')
        PHO = request.args.get('PHO')
        POR = request.args.get('POR')
        SAC = request.args.get('SAC')
        SAS = request.args.get('SAS')
        TOR = request.args.get('TOR')
        UTA = request.args.get('UTA')
        WAS = request.args.get('WAS')

        teams = [ATL, BOS, BKN, CHA, CHI, CLE, DAL, DET, GSW, HOU, IND,
                 LAC, LAL, MEM, MIA, MIL, NOP, OKC, ORL, PHO, POR, SAC,
                 SAS, TOR, UTA, WAS]
        input_team = []
        for team in teams:
            if team:
                input_team += [team]
    if len(input_team) < 1:
        input_team = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL',
                      'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL',
                      'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC',
                      'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR',
                      'UTA', 'WAS']

    # Grab data off database
    sql = "SELECT * FROM results"
    df = pd.read_sql_query(sql, g.db)
    df = df.drop('index', axis = 1)
    df = df[df['team'].isin(input_team)]
    df = df.sort_values('points', ascending = False)
    df = df.reset_index().drop('index', axis = 1)
    # Round values for display purposes
    for col in ['avg_pm','pred','points']:
        df[col] = np.round(df[col], 1)

    entries = [dict(lineup = df['lineup'][row],
               team = df['team'][row],
               opponent = df['opponent'][row],
               month = df['month'][row],
               day = df['dayofmonth'][row],
               base = df['avg_pm'][row],
               pred = df['pred'][row],
               points = df['points'][row]) for row in range(df.shape[0])]


    # Get dimensions of screen with Tkinter
    root = tk.Tk()
    scr_width = root.winfo_screenwidth()
    scr_height = root.winfo_screenheight()
    wide = scr_width * 0.4
    height = scr_height * 0.45


    # MAKE PLOT DATA
    pred = np.array(df['pred'])
    y_test = np.array(df['points'])
    avg_pm = np.array(df['avg_pm'])

    # Configure resources to include BokehJS inline in the document.
    js_resources = JS_RESOURCES.render(
        js_raw=INLINE.js_raw,
        js_files=INLINE.js_files
    )

    css_resources = CSS_RESOURCES.render(
        css_raw=INLINE.css_raw,
        css_files=INLINE.css_files
    )

    source = ColumnDataSource(data = dict(x = y_test,
                                          y = pred,
                                          lineup = np.array(df['lineup']),
                                          base = np.array(df['avg_pm']),
                                          month = np.array(df['month']),
                                          day = np.array(df['dayofmonth']),
                                          team = np.array(df['team']),
                                          opponent = np.array(df['opponent'])
                                          )
                              )


    # create a new plot with the tools above, and explicit ranges
    TOOLS="resize, pan, wheel_zoom, box_zoom, reset, hover"
    p = figure(tools = TOOLS, x_range=(-175,175), y_range=(-175, 175),
               plot_width = int(wide), plot_height = int(height))
    p.patch([0, 0, 5000, 5000], [0, 5000, 5000, 0],
            alpha =  0.3, color = (6, 110, 10))
    p.patch([0, 0, -5000, -5000],[0, -5000, -5000, 0],
            alpha = 0.3, color = (6, 110, 10))
    p.line(np.arange(-2000,2000,100), np.arange(-2000,2000,100),
           line_width = 2, color = "grey", legend = "Perfect Prediction")
    p.circle('x', 'base', source = source, radius=3,
             color = (0, 51, 127.5), alpha = 0.9, legend = "Base Model")
    p.circle('x', 'y', source = source, radius=3,
             color = (153, 0, 51), alpha = 0.9, legend = "New Model")
    p.legend.orientation = "top_left"
    p.xaxis.axis_label = 'Actual +/- (points/48min)'
    p.yaxis.axis_label = 'Predicted +/- (points/48min)'


    hover = p.select(dict(type=HoverTool))
    hover.point_policy = "follow_mouse"
    hover.tooltips = OrderedDict([("Lineup", "@lineup"),
                                  ("Team", "@team"),
                                  ("Opponent", "@opponent"),
                                  ("(Month, Day)", "(@month, @day)"),
                                  ("Actual", "@x"),
                                  ("Prediction", "@y"),
                                  ("Base Model", "@base")])


    plot_script, plot_div = components(p)

    # convert input_team to a string for html and then passed to img function
    input_team = ','.join(input_team)
    """
    # Debugging
    output_file("tt2.html")
    f = open('tfile', 'w')
    f.write(plot_script)
    f.close()
    show(p)
    """
    html = render_template("result.html", input_team = input_team,
                           entries = entries, wide = wide, height = height,
                           plot_script = plot_script, plot_div = plot_div,
                           js_resources=js_resources,
                           css_resources=css_resources)
    return encode_utf8(html)


# Not using anymore

@app.route('/img/<input_team>')
def img(input_team):
    input_team = list(input_team.split(','))
    print len(input_team)
    print input_team
    sql = "SELECT * FROM results"
    df = pd.read_sql_query(sql, g.db)
    df = df.drop('index', axis = 1)
    print 'Dataframe shape:', df.shape
    print list(df.columns)
    #con = connect_db()
    #cur = con.cursor()
    #cur.execute("SELECT * FROM results;")
    #con.commit()
    #print cur.fetchall()
    #con.close()
    p_df = df[df['team'].isin(input_team)].copy()
    print 'New Dataframe shape:', p_df.shape
    pred = np.array(p_df['pred'])
    y_test = np.array(p_df['points'])
    avg_pm = np.array(p_df['avg_pm'])


    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize = (10, 8))

    x1 = np.arange(0,180)
    y1 = np.zeros(180)+180
    x2 = np.arange(0,-180, -1)
    y2 = np.zeros(180)-180
    ax.plot(x1, y1)
    plt.fill_between(x1, y1, 0, color=(0.01,0.40,0.1), alpha = 0.25)
    plt.fill_between(x2, y2, 0, color=(0.01,0.40,0.1), alpha = 0.25)
    ax.scatter(y_test, avg_pm, color = (0,0.2,0.5),
               label = 'Base Model Predictions', s = 70, alpha = 1)
    ax.scatter(y_test, pred, color = (0.6,0.0,0.2),
               label = 'New Model Predictions',
               s = 70, alpha = 1)
    ax.plot(np.arange(-200, 200),np.arange(-200, 200), color = 'black',
               label = 'Perfect Prediction Line',
               lw = 3, alpha = 0.6, ls = 'dashed')
    #ax.plot(x,pred_y, label = 'Fit', lw = 5)
    ax.set_xlabel('Actual +/- (points/48 min)',fontsize = 14)
    ax.set_ylabel('Predicted +/- (points/48 min)', fontsize = 14)
    ax.set_title('Prediction Results', fontsize = 20)
    ax.set_xlim(-175,175)
    ax.set_ylim(-175,175)
    ax.legend(loc=2, fontsize = 12)
    ax.tick_params(labelsize =12)

    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response
