# Shamelessly copied from NeuralPlane Utils

import math

a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2-f)
pi = math.pi

def enu_to_ecef(xEast, yNorth, zUp, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)
    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda
    t = cos_lambda * zUp - sin_lambda * yNorth
    zd = sin_lambda * zUp + cos_lambda * yNorth
    xd = cos_phi * t - sin_phi * xEast
    yd = sin_phi * t + cos_phi * xEast
    x = xd + x0
    y = yd + y0
    z = zd + z0
    return x, y, z

def ecef_to_geodetic(x, y, z):
    # SAFETY CHECK: If coordinates are exploding, return dummy values
    # Python floats overflow around 1e308. We clip way before that.
    if abs(x) > 1e15 or abs(y) > 1e15 or abs(z) > 1e15:
        # Return 0,0,0 or just clamp. Returning 0 warns us something is wrong.
        return 0.0, 0.0, 0.0

    # ... (Original Code Below) ...
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    a = 6378137.0000    # earth radius in meters
    b = 6356752.3142    # earth semiminor in meters
    e = math.sqrt (1 - (b / a) ** 2)
    b2 = b * b
    e2 = e ** 2
    ep = e * (a / b)
    r = math.sqrt(x2 + y2)
    r2 = r * r
    E2 = a ** 2 - b ** 2
    F = 54 * b2 * z2
    G = r2 + (1 - e2) * z2 - e2 * E2

    # SAFETY CHECK 2: Prevent G from being zero or causing overflow
    if abs(G) < 1e-6:
        return 0.0, 0.0, 0.0

    c = (e2 * e2 * F * r2) / (G * G * G)
    s = (1 + c + math.sqrt(c * c + 2 * c)) ** (1 / 3)
    P = F / (3 * (s + 1 / s + 1) ** 2 * G * G)
    Q = math.sqrt(1 + 2 * e2 * e2 * P)
    ro = -(P * e2 * r) / (1 + Q) + math.sqrt((a * a / 2) * (1 + 1 / Q) - (P * (1 - e2) * z2) / (Q * (1 + Q)) - P * r2 / 2)
    tmp = (r - e2 * ro) ** 2
    U = math.sqrt(tmp + z2)
    V = math.sqrt(tmp + (1 - e2) * z2)
    zo = (b2 * z) / (a * V)
    height = U * (1 - b2 / (a * V))
    lat = math.atan((z + ep * ep *zo) / r)
    if x == 0:
        temp = pi/2 * (-1 if y < 0 else 1) # Handle x=0
    else:
        temp = math.atan(y / x)

    if x >=0 :
        long = temp
    elif (x < 0) & (y >= 0):
        long = pi + temp
    else :
        long = temp - pi
    lat0 = lat/(pi/180)
    lon0 = long/(pi/180)
    h0 = height
    return lat0, lon0, h0

def enu_to_geodetic(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref):
    x,y,z = enu_to_ecef(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref)
    return ecef_to_geodetic(x,y,z)