import matplotlib.pyplot as plt
import folium
import branca
import pandas as pd
import json
import numpy as np
from preproc import uscountygeo

# p_values = [0,0, 4.9e-6, 0, 0.0003, 0.0001, 0, 0.0004, 0, 0, 0.2, 0, 8e-5, 0.004, 0, 0, 0, 0, 0, 0, 0.5, 0.27, 0.5, 0.2, 0.006, 0.68, 0.002, 0.12, 0.3, 0.57, 0.5, 0.41, 0.55, 0.71, 0.56, 0.34]
# plt.plot(p_values)
# plt.title("Confirmed cases p-values")
# plt.xlabel("Confirmed cases lag")
# plt.ylabel("p-value")
# plt.show()
res = pd.read_csv("data/residuals1.csv")
#res = res.set_index("ID")
resids = np.array(res["Resid"])
resids = (resids + np.abs(np.min(resids)) + 0.02) / np.std(resids)
res["Resid"] = resids
print(resids)

ids = []
for x in res["ID"]:
    ids.append(str(int(x)))
res["ID"] = ids
res = res.set_index("ID")

res = res[res["Date"] == 0]
colorscale = branca.colormap.linear.RdBu_09.scale(np.log(0.01), np.log(500))

def style_function(feature):
    try:
        #print("Hello")
        county = str(int(feature['id'][-5:]))
        #print(county)
        data = np.log(res.loc[county, "Resid"])
        print(res.loc[county, "Resid"], data)

    except Exception as e:
        data = np.log(0.011)
        #print(data)
        #print("!!!!!")
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
    uscountygeo,
    'objects.us_counties_20m',
    style_function=style_function
).add_to(m)

m.save("result/result1.html")
