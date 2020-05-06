import numpy as np
from preproc.icu import icubydate_state, state_dict
import time
import json
import folium
import branca
import pandas as pd
import selenium.webdriver
from preproc import uscountygeo
import imageio
import json
from selenium import webdriver
from PIL import Image
import datetime

with open ("data/states-10m.json", "r") as f:
    state_json = json.load(f)

images = []
dates = icubydate_state.keys()
today = datetime.datetime.today()

for date in dates:
    if int(date.split("-")[1][1]) > today.month:
        continue
    print(date)

    colorscale = branca.colormap.linear.YlOrRd_09.scale(np.log(1e-1), np.log(60000))
    icu = icubydate_state[date]

    def style_function(feature):
        try:
            state = int(state_dict[feature["properties"]["name"]])
            data = icu.at[state, "allbed_mean"]
            data = np.log(1e-1) if data <= 0. else np.log(data)
        except Exception as e:
            data = np.log(1e-1)

        return {
            'fillOpacity': 0.5,
            'weight': 0,
            'fillColor': '#black' if data is None else colorscale(data)
        }

    m = folium.Map(
        location=[38, -95],
        tiles='cartodbpositron',
        zoom_start=4
    )

    folium.TopoJson(
        state_json,
        'objects.states',
        style_function=style_function
    ).add_to(m)

    m.save("result/icu-%s.html" % date)

    DRIVER = 'chromedriver'
    driver = webdriver.Chrome(DRIVER)
    "result/icu-%s.html" % "2020-03-19"
    driver.get('file:///C:/Users/Alexander/Desktop/COVID-19-Analysis/result/icu-%s.html' % date)
    screenshot = driver.save_screenshot('result/icu-%s.png' % date)
    driver.quit()
    foo = Image.open('result/icu-%s.png' % date)
    foo = foo.resize((600,400),Image.ANTIALIAS)
    foo.save('result/icu-%s.png' % date,quality=20,optimize=True)
    images.append(imageio.imread("result/icu-%s.png" % date))

imageio.mimsave('result/icu_viz.gif', images)
