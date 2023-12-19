#!/usr/bin/env python
"""
Zestaw funkcji do obliczeń związanych z opracowaniem wyrównania sieci kątowo-liniowej metodą pośredniczącą.

Ten moduł jest wolnym oprogramowaniem: można go rozpowszechniać i/lub modyfikować zgodnie z warunkami 
powszechnej licencji GNU.

Moduł jest rozpowszechniany z nadzieją, że będzie przydatny, ale BEZ ŻADNEJ GWARANCJI; 
nawet bez dorozumianej gwarancji PRZYDATNOŚCI DO OKREŚLONEGO CELU. 

Więcej szczegółów znajdziesz w Powszechnej Licencji Publicznej GNU.

"""
__author__ = "Dorota Marjanska"
__copyright__ = "Copyright 2023"
__credits__ = ["Dorota Marjanska" , "Joanna Kozuchowska", "Marcin Rajner"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Dorota Marjanska"
__email__ = "dorota.marjanska@pw.edu.pl"
__status__ = "Production"


from numpy import sqrt, arctan2, sin, cos, arccos, pi, ones, column_stack 

def check_points(pkt):
    """
    Jeśli do obliczeń została podana tablica jednowymiarowa, tj. pkt.shape = (2,),
    funkcja zwróci tablicę o wymiarach (1,2), aby obliczenia z tego pakietu mogły być wykonywane prawidłowo.

    Parameters
    ----------
    pkt : numpy.ndarray
        jedno lub dwuwymiarowa tablica numpy z punktami o wsp. XY

    Returns
    -------
    new_pkt : numpy.ndarray
        dwuwymiarowa tablica numpy z punktami o wsp. XY
    """    
    if len(pkt.shape) == 1:
        return pkt.reshape(1,-1)
    else:
        return pkt

def grad2rad(kat_w_gradach):
    """Funkcja przeliczająca kąt w gradach na kąt w radianach.
    
    Parameters
    ----------
    kat_w_gradach : int, float, numpy.ndarray
        wartość kąta w gradach, pojedyncza wartość lub tablica

    Returns
    -------
    kat_w_radianach : float, numpy.ndarray 
        wartość kąta w radianach, pojedyncza wartość lub tablica
    """    
    return kat_w_gradach / 200 * pi

def rad2grad(kat_w_radianach):
    """Funkcja przeliczająca kąt w radianach na kąt w gradach.
    
    Parameters
    ----------
    kat_w_radianach : int, float, numpy.ndarray
        wartość kąta w radianach, pojedyncza wartość lub tablica

    Returns
    -------
    kat_w_gradach : float, numpy.ndarray
    wartość kąta w gradach, pojedyncza wartość lub tablica
    """    
    return kat_w_radianach * 200 / pi

def odleglosc(pkt_pocz, pkt_konc):
    """Funkcja przeliczająca odległość płaską pomiędzy dwoma punktami.
    Działa dla jednej lub wielu obserwacji.

    Parameters
    ----------
    pkt_pocz :  numpy.ndarray
        współrzędne XY punktu początkowego
    pkt_konc :  numpy.ndarray 
        współrzędne XY punktu końcowego

    Returns
    -------
    odleglosc : numpy.ndarray
        odległość płaska między punktami pocz i konc
    """    
    # gdzie pkt_pocz = [x_p, y_p], pkt_konc = [x_k, y_k]
    pkt_pocz = check_points(pkt_pocz)
    pkt_konc = check_points(pkt_konc)
    
    # if len(pkt_pocz.shape) == 1:
    #     pkt_pocz = pkt_pocz.reshape(1,-1)
    # if len(pkt_konc.shape) == 1:
    #     pkt_konc = pkt_konc.reshape(1,-1)
        
    delta_x = pkt_konc[:,0] - pkt_pocz[:,0]
    delta_y = pkt_konc[:,1] - pkt_pocz[:,1]
    
    return sqrt(delta_x**2 + delta_y**2)

def azymut(pkt_pocz, pkt_konc):
    """
    Obliczenie azymutu między punktami początkowym i końcowym.
    Wynik w radianach.
    Działa dla jednej lub wielu obserwacji.
    
    Parameters
    ----------
    pkt_pocz : (numpy.ndarray)
            współrzędne XY punktu początkowego
    pkt_konc : (numpy.ndarray)
            współrzędne XY punktu końcowego

    Returns
    ----------
    azymut :    (numpy.ndarray)
            wartość azymutu/azymutów w radianach, bez uwzględnienia ćwiartki
    """    
    pkt_pocz = check_points(pkt_pocz)
    pkt_konc = check_points(pkt_konc)

    # if len(pkt_pocz.shape) == 1:
    #     pkt_pocz = pkt_pocz.reshape(1,-1)
    # if len(pkt_konc.shape) == 1:
    #     pkt_konc = pkt_konc.reshape(1,-1)

    deltaX = pkt_konc - pkt_pocz #dx i dy razem
    return (arctan2(deltaX[:,1], deltaX[:,0]))


# tutaj dla przykładu funkcja liczenia kąta NA PODSTAWIE AZYMUTÓW,
# jako argumenty podajemy współrzędne punktu lewego, centralnego, prawego
# X_lewy = np.array([x_l, y_l])...
def kat(X_lewy, X_centralny, X_prawy):
    """
    Funkcja oblicza kąt ze współrzędnych jako różnicę azymutów ramienia prawego i lewego. 

    Parameters
    ----------
    X_lewy : (numpy.ndarray)
        współrzędne XY punktu lewego
    X_centralny : (numpy.ndarray)
        współrzędne XY punktu centralnego
    X_prawy : (numpy.ndarray)
        współrzędne XY punktu prawego

    Returns
    ----------
    kat : (numpy.ndarray)
        roznica azymutow CP i CL [rad]
    """    
    X_lewy = check_points(X_lewy)
    X_centralny = check_points(X_centralny)
    X_prawy = check_points(X_prawy)
    
    # if len(X_lewy.shape) == 1:
    #     X_lewy = X_lewy.reshape(1,-1)
    # if len(X_centralny.shape) == 1:
    #     X_centralny = X_centralny.reshape(1,-1)
    # if len(X_prawy.shape) == 1:
    #     X_prawy = X_prawy.reshape(1,-1)
        
    #wartosc kata w radianach
    return (azymut(X_centralny, X_prawy) - azymut(X_centralny, X_lewy))


def wciecie_katowe(P1, kat1, P2, kat2):
    """
    Wcięcie kątowe na podstawie obserwacji w trójkącie. 
    Posiadając informacje na temat wsp. punktów P1 i P2 oraz kątów do punktu P3 obliczamy 
    - odległość P1-P2,
    - azymut P1-P3,
    - odległość P1-P3,
    - XY3 idąc z punktu P1 (czyli "prawego" patrząc z perspektywy punktu szukanego)
    
    Działa dla jednej lub wielu obserwacji. Za każdym razem trzeba ustawić odpowiednio obserwacje.

    Parameters
    ----------
    P1 : numpy.ndarray
        współrzędne XY punktu pierwszego
    kat1 : float, numpy.ndarray
        kąt z punktu "prawego" do punktu szukanego
    P2 : numpy.ndarray
        współrzędne XY punktu drugiego
    kat2 : float, numpy.ndarray
        kąt z punktu "lewego" do punktu szukanego

    Returns:
    ----------
        XY_szukane (numpy.ndarray): współrzędne XY punktu szukanego
    """  
    P1 = check_points(P1)
    P2 = check_points(P2)
    
    odl_12 = odleglosc(P1, P2)
    azX1X3 = azymut(P1, P2) - kat1
    odl_13 = odl_12/(sin(kat1 + kat2)) * sin(kat2)
    XY_szukane = column_stack((P1[:,0] + odl_13 * cos(azX1X3), P1[:,1] + odl_13 * sin(azX1X3)))
    
    # jeśli podajemy jeden punkt: spłaszaczamy wymiary do tablicy jednowymiarowej, żeby było wygodniej adresować do macierzy A
    if XY_szukane.shape[0] == 1:
        XY_szukane = XY_szukane.reshape(-1)
    
    return XY_szukane


def wciecie_liniowe(P1, odl1, P2, odl2):
    """
    Funkcja oblicza współrzędne punktu na podstawie obserwacji odległości w trójkącie i znanych współrzędnych dwóch punktów.
    Działa dla jednej lub wielu obserwacji. Za każdym razem trzeba ustawić odpowiednio obserwacje.

    Parameters
    ----------
    P1 : numpy.ndarray
        współrzędne XY punktu pierwszego
    odl1 : float, numpy.ndarray
        odległość z punktu "prawego" do punktu szukanego
    P2 : numpy.ndarray
        współrzędne XY punktu drugiego
    odl2 : float, numpy.ndarray
        odległość z punktu "lewego" do punktu szukanego

    Returns
    ----------
    XY_szukane : (numpy.ndarray)
        współrzędne XY punktu szukanego
    """
    P1 = check_points(P1)
    P2 = check_points(P2)
    
    odl3 = odleglosc(P1, P2)
    katy = column_stack((arccos((-odl2**2 + odl1**2 + odl3**2) / (2 * odl1 * odl3)), arccos((-odl1**2 + odl2**2 + odl3**2) / (2 * odl2 * odl3))))
    XY_szukane = wciecie_katowe(P1, katy[:,0], P2, katy[:,1])
    
    return XY_szukane

## returns entries in order for:
##
def dme_kat(Xl, Xc, Xp):
    """
    Funkcja zwraca elementy do macierzy kształtu (współczynników przy niewiadomych) dla obserwacji odległości.
    Następnie w zależności od tego, czy obserwacja dotyczy któregoś z punktów szukanych czy stałych, wybiera się elementy do macierzy kształtu.
    Działa dla jednej lub więcej obserwacji.

    Parameters
    ----------
    Xl : numpy.ndarray
        współrzędne XY punktu lewego
    Xc : numpy.ndarray
        współrzędne XY punktu centralnego
    Xp : numpy.ndarray
        współrzędne XY punktu prawego 
        
    Returns
    ----------
    dme : numpy.ndarray
        (wszystkie możliwe) elementy do macierzy kształtu w kolejności: dXl dYl dXp dYp dXc dYc 
    """    
    
    Xl = check_points(Xl)
    Xc = check_points(Xc)
    Xp = check_points(Xp) 
    
    # if len(Xl.shape) == 1:
    #     Xl = Xl.reshape(1,-1)
    # if len(Xc.shape) == 1:
    #     Xc = Xc.reshape(1,-1)
    # if len(Xp.shape) == 1:
    #     Xp = Xp.reshape(1,-1)
    
    dcl = odleglosc(Xc, Xl)
    dcp = odleglosc(Xc, Xp)
    
    dme = column_stack(( (Xl[:, 1] - Xc[:, 1]) / dcl**2, #  dx(CL) / d^2
                        -(Xl[:, 0] - Xc[:, 0]) / dcl**2, # -dy(CL) / d^2
                        -(Xp[:, 1] - Xc[:, 1]) / dcp**2, # -dx(CP) / d^2
                        (Xp[:, 0] - Xc[:, 0]) / dcp**2)) #  dy(CP) / d^2
    
    dme5 = (-dme[:, 0] - dme[:, 2]) # -(dx(CL) / d^2) - (-dx(CP) / d^2)
    dme6 = (-dme[:, 1] - dme[:, 3]) # -(-dy(CL) / d^2) - (-dy(CP) / d^2)
    dme = column_stack((dme, dme5, dme6))
    
    # jeśli podajemy jeden punkt: spłaszaczamy wymiary do tablicy jednowymiarowej, żeby było wygodniej adresować do macierzy A
    if dme.shape[0] == 1:
        dme = dme.reshape(-1)
    
    return dme

##dme design matrix entries for bearing
def dme_kierunek(Xp, Xk):
    """
    Funkcja zwraca elementy do macierzy kształtu (współczynników przy niewiadomych) dla obserwacji kierunków.
    Następnie w zależności od tego, czy obserwacja dotyczy któregoś z punktów szukanych czy stałych, wybiera się elementy do macierzy kształtu.
    Działa dla jednej lub więcej obserwacji.

    Parameters
    ----------
    Xp : numpy.ndarray
        współrzędne XY punktu początkowego
    Xk : numpy.ndarray
        współrzędne XY punktu końcowego

    Returns
    ----------
    dme : numpy.ndarray
        (wszystkie możliwe) elementy do macierzy kształtu w kolejności: dXP, dYP, dXK, dYK 
    """
    Xp = check_points(Xp) 
    Xk = check_points(Xk) 
    
    # if len(Xp.shape) == 1:
    #     Xp = Xp.reshape(1,-1)
    # if len(Xk.shape) == 1:
    #     Xk = Xk.reshape(1,-1)
    
    d = odleglosc(Xp, Xk)

    npoints = max([Xp.shape[0], Xk.shape[0]])
    dme = column_stack(( (Xk[:,1] - Xp[:,1])/d**2, #  dy(PK) / d^2
                        -(Xk[:,0] - Xp[:,0])/d**2, # -dx(PK) / d^2
                        -(Xk[:,1] - Xp[:,1])/d**2, # -dy(PK) / d^2
                        (Xk[:,0] - Xp[:,0])/d**2,  #  dx(PK) / d^2
                        -1 * ones(npoints) ))        # -1 do każdej obserwacji kierunku
    
    # jeśli podajemy jeden punkt: spłaszaczamy wymiary do tablicy jednowymiarowej, żeby było wygodniej adresować do macierzy A
    if dme.shape[0] == 1:
        dme = dme.reshape(-1)
    
    return dme


def dme_odleglosc(Xp, Xk):
    """
    Funkcja zwraca elementy do macierzy kształtu (współczynników przy niewiadomych) dla obserwacji odległości.
    Następnie w zależności od tego, czy obserwacja dotyczy któregoś z punktów szukanych czy stałych, wybiera się elementy do macierzy kształtu.
    Działa dla jednej lub więcej obserwacji.

    Parameters
    ----------
    Xp : numpy.ndarray
        współrzędne XY punktu początkowego
    Xk : numpy.ndarray
        współrzędne XY punktu końcowego

    Returns
    ----------
    dme :  numpy.ndarray
        (wszystkie możliwe) elementy do macierzy kształtu w kolejności: dXP, dYP, dXK, dYK 
    """
    
    Xp = check_points(Xp) 
    Xk = check_points(Xk) 
    
    # if len(Xp.shape) == 1:
    #     Xp = Xp.reshape(1,-1)
    # if len(Xk.shape) == 1:
    #     Xk = Xk.reshape(1,-1)
    
    az = azymut(Xp, Xk)
    dme = column_stack((-cos(az), -sin(az), cos(az), sin(az)))
    
    # jeśli podajemy jeden punkt: spłaszaczamy wymiary do tablicy jednowymiarowej, żeby było wygodniej adresować do macierzy A
    if dme.shape[0] == 1:
        dme = dme.reshape(-1)
    
    return dme

if __name__ == '__main__':
    # przyklad dzialania funkcji
    import numpy as np
    # po dwie obserwacje roznego rodzaju
    X_P = np.array([[100, 100], [200, 300]])
    X_K = np.array([[100, 300], [300, 200]])
    X_C = np.array([[300, 100], [100, 100]])
    
    print('odleglosci = ', odleglosc(X_K, X_P))
    print('azymuty = ', azymut(X_P, X_K))
    print('azymuty w gradach = ', rad2grad(azymut(X_P, X_K)))
    
    kat_test = wciecie_katowe(X_P[0,:], kat(X_C[0,:], X_P[0,:], X_K[0,:]), X_K[0,:], kat(X_P[0,:], X_K[0,:], X_C[0,:]))
    print('wciecie katowe na punkt X_C (jedna obs) = ', kat_test, kat_test.shape)
    
    kat_test2 = wciecie_katowe(X_P, kat(X_C, X_P, X_K), X_K, kat(X_P, X_K, X_C))
    print('wciecie katowe na punkt X_C (dwie obs) = ', kat_test)
    
    lin_test = wciecie_liniowe(X_P[0,:], odleglosc(X_P[0,:], X_C[0,:]), X_K[0,:], odleglosc(X_K[0,:], X_C[0,:]))
    print('wciecie liniowe na punkt X_C (jedna obs) = ', lin_test)
    
    lin_test2 = wciecie_liniowe(X_P, odleglosc(X_P, X_C), X_K, odleglosc(X_K, X_C))
    print('wciecie liniowe na punkty X_C (dwie obs) = ', lin_test2)
    
    
    wspolczynniki_obs_odleglosci = dme_odleglosc(X_K, X_P)
    wspolczynniki_obs_kierunku = dme_kierunek(X_P, X_K)
    wspolczynniki_obs_kata = dme_kat(X_P, X_C, X_K)
    print(wspolczynniki_obs_odleglosci, wspolczynniki_obs_odleglosci.shape)
    print(wspolczynniki_obs_kierunku, wspolczynniki_obs_kierunku.shape)
    print(wspolczynniki_obs_kata, wspolczynniki_obs_kata.shape)
    
    A = np.zeros((7, 5))
    # przyklad uzupelnienia macierzy A dwoma obserwacjami kierunku, TYLKO SPRAWDZENIE DZIALANIA FUNKCJI!!!
    # szukane XP i XK, wybieram tylko jedna obserwacje z dwoch w danych startowych 
    A[0, :] = wspolczynniki_obs_kierunku[0] # 5 wspolczynnikow, kierunek XP do XK, nie musze nic zmieniac
    A[1, 0:4] = wspolczynniki_obs_kata[0, :4] # kat XP (lewy), XC (centralny), XK (prawy), wspolczynniki zwracane w kolejnosci lewy-prawy-centralny
    A[2, [2, 3, 0, 1]] = wspolczynniki_obs_odleglosci[0] # obserwacja odleglosci od XK do XP, wiec musze wziac wspolczynniki w odpowiedniej kolejnosci
    
    print(A)
