from adjacency import adj_dict
from preproc import covid19bydate
import pandas as pd
from datetime import date, timedelta
from shapely import geometry
import json
import numpy as np
import folium
import branca
import statsmodels.api as sm
from preproc import uscountygeo

# Make time window (15 days)
window = 20
start = date(2020, 3, 10)
date_list = [ str(start + timedelta(days=1 * i)) for i in range(window) ]
d_list = [ start + timedelta(days=i) for i in range(window) ]
frames = []
sdate = str(start)
covid19 = covid19bydate[sdate]
counties = list(covid19.index)

for lag in range(10):
    print(lag)
    # Format the data s that it fits with the R GLS package
    for k in range(window):
        date = d_list[k]
        ld = date - timedelta(days = lag)
        lag_date = str(ld)
        sdate = date_list[k]
        print(sdate)
        # Select time lag
        lag = 50
        length = len(covid19bydate[sdate]["death"])

        data = {"Death": covid19bydate[sdate]["death"], "Latitude": np.zeros(length), "Longitude": np.zeros(length)}
        data = pd.DataFrame(data, index = covid19bydate[sdate].index)


        # Get the latitude and longitude of each county
        with open("data\counties.json", "r") as f:
            county_geo = json.load(f)

        for geo in county_geo["features"]:
            #print(geo["geometry"]["coordinates"])
            county_id = str(int(geo["properties"]["GEO_ID"][-5:]))
            lat = 0
            lng = 0
            count = 0
            if geo["geometry"]["type"] == "Polygon":
                for arc in geo["geometry"]["coordinates"]:
                    for coord in arc:
                        count += 1
                        lat += coord[0]
                        lng += coord[1]
                lat /= count
                lng /= count
            else:
                for poly in geo["geometry"]["coordinates"]:
                    for arc in geo["geometry"]["coordinates"]:
                        for lisr in arc:
                            for coord in lisr:
                                count += 1
                                lat += coord[0]
                                lng += coord[1]
                lat /= count
                lng /= count
            try:
                if county_id in data.index:
                    data.loc[county_id, "Latitude"] = lat
                    data.loc[county_id, "Longitude"] = lng
            except:
                pass



        '''
        Add Major Hubs
        - Atlanta: 13121
        - NY: 36061
        - DC: 11001
        - LA: 6037
        - SF: 6075
        - Seattle: 53033
        - Chicago: 17031
        - Dallas: 48113
        - Miami: 12086
        - Boston: 25025
        - Detroit: 26163
        - Denver: 8031
        - Portland: 41051
        - Philadelphia: 42101
        '''
        # for i in counties:
        #     print(i)
        #     for j in counties:
        #         if j in adj_dict[i]:
        #             title = i + "_" + j
        #             count = covid19bydate[sdate].loc[j, "confirmed"]
        #             if count:
        #                 data[title] = np.zeros(len(counties))
        #                 data.loc[i, title] = covid19bydate[sdate].loc[j, "confirmed"]

        for i in counties:
            title = i + "_NYC"
            count = covid19bydate[lag_date].loc["36061", "confirmed"]
            data[title] = np.zeros(len(counties))
            data.loc[i, title] = covid19bydate[lag_date].loc["36061", "confirmed"]
        data = pd.DataFrame(data)
        frames.append(data)

    # Save dataframe
    data = pd.concat(frames)

    Y = data["Death"]
    X = data.drop(columns = ["Death"])

    mod = sm.OLS(Y, X)
    lm = mod.fit()
    x = lm.summary()

    a = str(x)
    s = a.split("\n")
    county_dict = {}
    for line in s:
        try:
            t = line.split("\t")[0]
            p_list = []
            for element in t.split(" "):
                if element != "":
                    p_list.append(element)
            if p_list[0].split("_")[0][0] in "0987654321":
                county_dict[p_list[0].split("_")[0]] = float(p_list[4])
        except:
            pass
    #print(county_dict)
    mx = 0
    for key in county_dict.keys():
        if county_dict[key] > mx:
            mx = county_dict[key]

    colorscale = branca.colormap.linear.YlOrRd_09.scale(0, mx)
    colorscale.caption = 'significance of NYC cases (1 - p)'

    def style_function(feature):
        county = str(int(feature['id'][-5:]))
        try:
            data = 1 - county_dict[county]
            #data = np.log(1e-1) if data <= 0. else np.log(data)
        except Exception as e:
            data = 0
        return {
            'fillOpacity': 0.5,
            'weight': 0,
            'fillColor': '#black' if data is None else colorscale(data)
        }

    m = folium.Map(
        location=[38, -95],
        tiles='cartodbpositron',
        zoom_start=5
    )

    folium.TopoJson(
        uscountygeo,
        'objects.us_counties_20m',
        style_function=style_function
    ).add_to(m)

    colorscale.add_to(m)

    m.save("p_value/pvalue_j=NYC_tau=" + str(lag) + ".html")
