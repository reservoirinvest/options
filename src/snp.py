import pandas as pd

dls = "http://www.cboe.com/products/weeklys-options/available-weeklys"
data = pd.read_html(dls)


