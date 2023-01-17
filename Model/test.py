import numpy as np

# \multirow{2}{*}{CL}   
# & S    & 29.99          & 33.69           & 31.95             & 1.04        & 0.01    & 19.34 \\
# & M    & 30.06         & 34.13           & 32.04             & 26.73       & 23.21   & 29.23 \\ \midrule[0.3pt]
# \multirow{2}{*}{ML}  
# & S    & 29.41          & 28.88           & 25.62             & 28.88        & 25.62   & 27.68  \\
# & M  & \textbf{30.76} & 32.29           & 26.66             &\textbf{32.29} & 26.66  & 29.73   \\ \midrule[0.3pt]
# \multirow{2}{*}{CML}  
# & S   & 25.21          & 36.46           & 27.08             & 20.13     & 22.51          & 26.28       \\
# & M   & 28.16          & \textbf{38.95}  & \textbf{33.83}    &28.01      & \textbf{29.18} & \textbf{31.63} \\ \midrule

a = [29.99, 29.41, 25.21]
b = [30.06, 30.76, 28.16]

c = [33.69, 28.88, 36.46]
d = [34.13, 32.29, 38.95]

e = [31.95, 25.62, 27.08]
f = [32.04, 26.66, 33.83]

j = [1.04, 28.88, 20.13]
k = [26.73,32.29, 28.01]

j1 = [0.01, 25.62, 22.51]
k1 = [23.21,26.66, 29.18]

j2 = [19.34,27.68, 26.28]
k2 = [29.23,29.73, 31.63]

print(round(np.average(a),2))
print(round(np.average(b),2))
print(round(np.average(c),2))
print(round(np.average(d),2))

print(round(np.average(e),2))
print(round(np.average(f),2))
print(round(np.average(j),2))
print(round(np.average(k),2))

print(round(np.average(j1),2))
print(round(np.average(k1),2))
print(round(np.average(j2),2))
print(round(np.average(k2),2))