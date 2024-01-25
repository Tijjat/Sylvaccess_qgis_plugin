# -*- coding: utf8 -*-
"""
Software: Sylvaccess
File: sylvaccess_cython.pyx
Copyright (C) Sylvain DUPIRE 2021
Authors: Sylvain DUPIRE
Contact: sylvain.dupire@inrae.fr
Version: 3.5.1
Date: 2021/12/17
License :  GNU-GPL V3

-----------------------------------------------------------------------
Français: (For english see above)

Ce fichier fait partie de Sylvaccess qui est un programme informatique 
servant à cartographier automatiquement les forêts accessibles en fonction de 
différents modes d'exploitations (skidder, porteur, débardage par câble aérien). 

Ce logiciel est un logiciel libre ; vous pouvez le redistribuer ou le modifier 
suivant les termes de la GNU General Public License telle que publiée par la 
Free Software Foundation ; soit la version 3 de la licence, soit (à votre gré) 
toute version ultérieure.
Sylvaccess est distribué dans l'espoir qu'il sera utile, mais SANS AUCUNE 
GARANTIE ; sans même la garantie tacite de QUALITÉ MARCHANDE ou d'ADÉQUATION 
à UN BUT PARTICULIER. Consultez la GNU General Public License pour plus de 
détails.
Vous devez avoir reçu une copie de la GNU General Public License en même temps 
que Sylvaccess ; si ce n'est pas le cas, consultez <http://www.gnu.org/licenses>.

-----------------------------------------------------------------------
English

This file is part of Sylvaccess which is a computer program whose purpose 
is to automatically map forest accessibility according to different forest 
operation systems (skidder, forwarder, cable yarding)..

Sylvaccess is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Foobar is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Sylvaccess.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
cimport numpy as np
import cython
cimport cython

##############################################################################################################################################
### Fonctions generales
##############################################################################################################################################
cdef extern from "math.h":
    double acos(double)
    double atan(double)
    double asin(double)
    double sqrt(double)
    double cos(double)
    double fabs(double)
    double log(double)
    double cos(double)
    double tan(double)
    double sin(double) 
    double sinh(double)
    double cosh(double)
    int floor(double)
    int ceil(double)    
    double atan2(double a, double b)
    double degrees(double)
    bint isnan(double x)


cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

cdef extern from "Python.h":
    Py_ssize_t PyList_Size(object) except -1   #Equivalent de len(list)
    int PyList_Append(object, object) except -1 #Equivalent de list.append()
    object PyList_GetSlice(object, Py_ssize_t low, Py_ssize_t high) #Equivalent de list[low:high]
    int PyList_Sort(object) except -1 #Equivalent de list.sort()

ctypedef np.int_t dtype_t
ctypedef np.float_t dtypef_t
ctypedef np.int8_t dtype8_t
ctypedef np.uint8_t dtypeu8_t
ctypedef np.uint16_t dtypeu16_t
ctypedef np.int16_t dtype16_t
ctypedef np.int32_t dtype32_t
ctypedef np.int64_t dtype64_t
ctypedef np.float32_t dtypef32_t

# Retourne le maximum d'un array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_array(np.ndarray[dtype_t, ndim=1] a):
    cdef int max_a = a[0]
    cdef unsigned int item = 1
    for item from 1 <= item < a.shape[0]:
        if a[item] > max_a:max_a = a[item]
    return max_a

# Retourne le maximum d'un array de float
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double max_array_f(np.ndarray[dtypef_t, ndim=1] a):
    cdef double max_a = a[0]
    cdef unsigned int item = 1
    for item from 1 <= item < a.shape[0]:
        if a[item] > max_a:max_a = a[item]
    return max_a

# Retourne la somme d'un array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double sum_array_f(np.ndarray[dtypef_t, ndim=1] a):
    cdef double summ = a[0]
    cdef unsigned int item = 1
    for item from 1 <= item < a.shape[0]:
        summ += item
    return summ

# Definie la fonction arcsinus hyperbolique
cdef inline double asinh(double x):return log(x+sqrt(1+x*x))
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b
cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double square(double x):return x*x
cdef inline int cint(double x):return int(x)
#cdef inline int isnan(value):return 1 if value != value else 0


cdef double g = 9.80665,pi=3.141592653589793

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype8_t,ndim=2] copy_int8_array(np.ndarray[dtype8_t,ndim=2] arraystock,np.ndarray[dtype8_t,ndim=2] arraytocopy, int nline, int ncol):
    cdef int i,j
    for i from 0<=i<nline:
        for j from 0<=j<ncol:            
            arraystock[i,j] = arraytocopy[i,j]
    return arraystock

# Renvoie la somme des valeurs d'un array d'entier 8
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long int8_array_sum(np.ndarray[dtype8_t, ndim=2] matrice):
    cdef long somme = 0
    cdef unsigned int y = 0
    cdef unsigned int x = 0
    for y from 0<=y<matrice.shape[0]:
        for x from 0<=x<matrice.shape[1]:
            somme += matrice[y,x]
    return somme

# Renvoie la somme des valeurs d'un array d'entier 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long int_array_sum(np.ndarray[dtype_t, ndim=2] matrice):
    cdef long somme = 0
    cdef unsigned int y = 0
    cdef unsigned int x = 0
    for y from 0<=y<matrice.shape[0]:
        for x from 0<=x<matrice.shape[1]:
            somme += matrice[y,x]
    return somme


##############################################################################################################################################
### Fonctions d'analyse spatiale
##############################################################################################################################################
#Calcule la pente a partir d'un MNT et de la taille de Cellule
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtypef32_t,ndim=2] pente(np.ndarray[dtypef_t,ndim=2] raster_mnt,double Csize,double nodata):
    """
    Calcule la pente en % a partir d'un raster MNT et d'une taille de cellule donnee
    -----
    Inputs: Raster_MNT en float, Cell size; Nodata Value
    -----
    Output : Raster de pente en %
    """
    cdef unsigned int nline= raster_mnt.shape[0]
    cdef unsigned int ncol = raster_mnt.shape[1]
    cdef np.ndarray[dtypef32_t,ndim=2] pente = np.zeros_like(raster_mnt,dtype=np.float32)
    cdef unsigned int x=1
    cdef unsigned int y=1
    cdef double a,b,c,d,e,f,g,h,i
    cdef double dz_dx,dz_dy
    # Grille sans les bordures
    for y from 1 <= y < nline-1:
        for x from 1 <= x < ncol-1:
            e = raster_mnt[y,x] 
            if e > nodata:
                a = raster_mnt[y-1,x-1]
                if a==nodata:a=e
                b = raster_mnt[y-1,x] 
                if b==nodata:b=e
                c = raster_mnt[y-1,x+1]
                if c==nodata:c=e
                d = raster_mnt[y,x-1]    
                if d==nodata:d=e
                f = raster_mnt[y,x+1]
                if f==nodata:f=e
                g = raster_mnt[y+1,x-1]  
                if g==nodata:g=e
                h = raster_mnt[y+1,x] 
                if h==nodata:h=e
                i = raster_mnt[y+1,x+1]
                if i==nodata:i=e                            
                dz_dx = float(c+2*f+i-(a+2*d+g))/float(8*Csize)
                dz_dy = float(g+2*h+i-(a+2*b+c))/float(8*Csize)
                pente[y,x]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100
            else: pente[y,x]=nodata
    # Coin superieur gauche
    if raster_mnt[0,0]>nodata:
        e = raster_mnt[0,0]
        f = raster_mnt[0,1]
        if f==nodata:f=e
        h = raster_mnt[1,0]
        if h==nodata:h=e
        i = raster_mnt[1,1]
        if i==nodata:i=e
        dz_dx = float(f+i-(e+h))/float(2*Csize)
        dz_dy = float(h+i-(d+f))/float(2*Csize)
        pente[0,0]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100      
    else: pente[0,0]=nodata
    # Coin inferieur gauche    
    if raster_mnt[nline-1,0]>nodata:
        e = raster_mnt[nline-1,0]
        b = raster_mnt[nline-2,0]
        if b==nodata:b=e
        c = raster_mnt[nline-2,1]
        if c==nodata:c=e
        f = raster_mnt[nline-1,1]
        if f==nodata:f=e
        dz_dx = float(c+f-(b+e))/float(2*Csize)
        dz_dy = float(e+f-(b+c))/float(2*Csize)
        pente[nline-1,0]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100       
    else: pente[nline-1,0]=nodata 
    # Coin superieur droite
    if raster_mnt[0,ncol-1]>nodata:
        e = raster_mnt[0,ncol-1]
        d = raster_mnt[0,ncol-2]
        if d==nodata:d=e
        g = raster_mnt[1,ncol-2]
        if g==nodata:g=e
        h = raster_mnt[1,ncol-1]
        if h==nodata:h=e
        dz_dx = float(e+h-(d+g))/float(2*Csize)
        dz_dy = float(g+h-(d+e))/float(2*Csize)
        pente[0,ncol-1]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100      
    else: pente[0,ncol-1]=nodata
    # Coin inferieur droite
    if raster_mnt[nline-1,ncol-1]>nodata:
        e = raster_mnt[nline-1,ncol-1]
        a = raster_mnt[nline-2,ncol-2]
        if a==nodata:a=e
        d = raster_mnt[nline-1,ncol-2]
        if d==nodata:d=e
        b = raster_mnt[nline-2,ncol-1]
        if b==nodata:b=e
        dz_dx = float(e+b-(d+a))/float(2*Csize)
        dz_dy = float(d+e-(a+b))/float(2*Csize)
        pente[nline-1,ncol-1]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100       
    else: pente[nline-1,ncol-1]=nodata
    # Pour premiere ligne
    x=1
    for x from 1 <= x < ncol-1:
        e = raster_mnt[0,x] 
        if e > nodata:            
            d = raster_mnt[0,x-1]    
            if d==nodata:d=e
            f = raster_mnt[0,x+1]
            if f==nodata:f=e
            g = raster_mnt[1,x-1]  
            if g==nodata:g=e
            h = raster_mnt[1,x] 
            if h==nodata:h=e
            i = raster_mnt[1,x+1]
            if i==nodata:i=e                            
            dz_dx = float(f+i-(d+g))/float(4*Csize)
            dz_dy = float(g+h+i-(d+e+f))/float(3*Csize)
            pente[0,x]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100
        else: pente[0,x]=nodata
    # Pour derniere ligne
    x=1
    for x from 1 <= x < ncol-1:
        e = raster_mnt[nline-1,x] 
        if e > nodata:            
            d = raster_mnt[nline-1,x-1]    
            if d==nodata:d=e
            f = raster_mnt[nline-1,x+1]
            if f==nodata:f=e
            a = raster_mnt[nline-2,x-1]  
            if a==nodata:a=e
            b = raster_mnt[nline-2,x] 
            if b==nodata:b=e
            c = raster_mnt[nline-2,x+1]
            if c==nodata:c=e                            
            dz_dx = float(f+c-(d+a))/float(4*Csize)
            dz_dy = float(d+e+f-(a+b+c))/float(3*Csize)
            pente[nline-1,x]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100
        else: pente[nline-1,x]=nodata
    # Pour premiere colonne
    y=1
    for y from 1 <= y < nline-1:
        e = raster_mnt[y,0] 
        if e > nodata:            
            b = raster_mnt[y+1,0]    
            if b==nodata:b=e
            c = raster_mnt[y+1,1]
            if c==nodata:c=e
            f = raster_mnt[y,1]  
            if f==nodata:f=e
            h = raster_mnt[y+1,0] 
            if h==nodata:h=e
            i = raster_mnt[y+1,1]
            if i==nodata:i=e                            
            dz_dx = float(c+f+i-(b+e+h))/float(3*Csize)
            dz_dy = float(h+i-(b+c))/float(4*Csize)
            pente[y,0]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100
        else: pente[y,0]=nodata
    # Pour derniere colonne
    y=1
    for y from 1 <= y < nline-1:
        e = raster_mnt[y,ncol-1] 
        if e > nodata:            
            a = raster_mnt[y-1,ncol-2]    
            if a==nodata:a=e
            b = raster_mnt[y-1,ncol-1]
            if b==nodata:b=e
            d = raster_mnt[y,ncol-2]  
            if d==nodata:d=e
            g = raster_mnt[y+1,ncol-2] 
            if g==nodata:g=e
            h = raster_mnt[y+1,ncol-1]
            if h==nodata:h=e                            
            dz_dx = float(b+e+h-(a+d+g))/float(3*Csize)
            dz_dy = float(h+g-(b+a))/float(4*Csize)
            pente[y,ncol-1]= sqrt(dz_dx*dz_dx+dz_dy*dz_dy)*100
        else: pente[y,ncol-1]=nodata
    return pente 

#Calcule la pente a partir d'un MNT et de la taille de Cellule
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtypef_t,ndim=2] exposition(np.ndarray[dtypef_t,ndim=2] raster_mnt,double Csize,double nodata):
    """
    Calcule la expo en % a partir d'un raster MNT et d'une taille de cellule donnee
    -----
    Inputs: Raster_MNT en float, Cell size; Nodata Value
    -----
    Output : Raster de expo en %
    """
    cdef unsigned int nline= raster_mnt.shape[0]
    cdef unsigned int ncol = raster_mnt.shape[1]
    cdef np.ndarray[dtypef_t,ndim=2] expo = np.zeros_like(raster_mnt,dtype=np.float)
    cdef unsigned int x=1
    cdef unsigned int y=1
    cdef double a,b,c,d,e,f,g,h,i
    cdef double dz_dx,dz_dy,expo1
    # Grille sans les bordures
    for y from 1 <= y < nline-1:
        for x from 1 <= x < ncol-1:
            e = raster_mnt[y,x] 
            if e > nodata:
                a = raster_mnt[y-1,x-1]
                if a==nodata:a=e
                b = raster_mnt[y-1,x] 
                if b==nodata:b=e
                c = raster_mnt[y-1,x+1]
                if c==nodata:c=e
                d = raster_mnt[y,x-1]    
                if d==nodata:d=e
                f = raster_mnt[y,x+1]
                if f==nodata:f=e
                g = raster_mnt[y+1,x-1]  
                if g==nodata:g=e
                h = raster_mnt[y+1,x] 
                if h==nodata:h=e
                i = raster_mnt[y+1,x+1]
                if i==nodata:i=e                            
                dz_dx = float(c+2*f+i-(a+2*d+g))/float(8*Csize)
                dz_dy = float(g+2*h+i-(a+2*b+c))/float(8*Csize)
                expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
                if expo1<0.:
                    expo[y,x]= 90.0 - expo1
                elif expo1>90.:
                    expo[y,x]= 360.0 - expo1 + 90.0
                else:
                    expo[y,x]= 90.0 - expo1
            else: expo[y,x]=nodata
    # Coin superieur gauche
    if raster_mnt[0,0]>nodata:
        e = raster_mnt[0,0]
        f = raster_mnt[0,1]
        if f==nodata:f=e
        h = raster_mnt[1,0]
        if h==nodata:h=e
        i = raster_mnt[1,1]
        if i==nodata:i=e
        dz_dx = float(f+i-(e+h))/float(2*Csize)
        dz_dy = float(h+i-(d+f))/float(2*Csize)
        expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
        if expo1<0.:
            expo[0,0]= 90.0 - expo1
        elif expo1>90.:
            expo[0,0]= 360.0 - expo1 + 90.0
        else:
            expo[0,0]= 90.0 - expo1    
    else: expo[0,0]=nodata
    # Coin inferieur gauche    
    if raster_mnt[nline-1,0]>nodata:
        e = raster_mnt[nline-1,0]
        b = raster_mnt[nline-2,0]
        if b==nodata:b=e
        c = raster_mnt[nline-2,1]
        if c==nodata:c=e
        f = raster_mnt[nline-1,1]
        if f==nodata:f=e
        dz_dx = float(c+f-(b+e))/float(2*Csize)
        dz_dy = float(e+f-(b+c))/float(2*Csize)
        expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
        if expo1<0.:
            expo[nline-1,0]= 90.0 - expo1
        elif expo1>90.:
            expo[nline-1,0]= 360.0 - expo1 + 90.0
        else:
            expo[nline-1,0]= 90.0 - expo1          
    else: expo[nline-1,0]=nodata 
    # Coin superieur droite
    if raster_mnt[0,ncol-1]>nodata:
        e = raster_mnt[0,ncol-1]
        d = raster_mnt[0,ncol-2]
        if d==nodata:d=e
        g = raster_mnt[1,ncol-2]
        if g==nodata:g=e
        h = raster_mnt[1,ncol-1]
        if h==nodata:h=e
        dz_dx = float(e+h-(d+g))/float(2*Csize)
        dz_dy = float(g+h-(d+e))/float(2*Csize)
        expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
        if expo1<0.:
            expo[0,ncol-1]= 90.0 - expo1
        elif expo1>90.:
            expo[0,ncol-1]= 360.0 - expo1 + 90.0
        else:
            expo[0,ncol-1]= 90.0 - expo1           
    else: expo[0,ncol-1]=nodata
    # Coin inferieur droite
    if raster_mnt[nline-1,ncol-1]>nodata:
        e = raster_mnt[nline-1,ncol-1]
        a = raster_mnt[nline-2,ncol-2]
        if a==nodata:a=e
        d = raster_mnt[nline-1,ncol-2]
        if d==nodata:d=e
        b = raster_mnt[nline-2,ncol-1]
        if b==nodata:b=e
        dz_dx = float(e+b-(d+a))/float(2*Csize)
        dz_dy = float(d+e-(a+b))/float(2*Csize)
        expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
        if expo1<0.:
            expo[nline-1,ncol-1]= 90.0 - expo1
        elif expo1>90.:
            expo[nline-1,ncol-1]= 360.0 - expo1 + 90.0
        else:
            expo[nline-1,ncol-1]= 90.0 - expo1 
    else: expo[nline-1,ncol-1]=nodata
    # Pour premiere ligne
    x=1
    for x from 1 <= x < ncol-1:
        e = raster_mnt[0,x] 
        if e > nodata:            
            d = raster_mnt[0,x-1]    
            if d==nodata:d=e
            f = raster_mnt[0,x+1]
            if f==nodata:f=e
            g = raster_mnt[1,x-1]  
            if g==nodata:g=e
            h = raster_mnt[1,x] 
            if h==nodata:h=e
            i = raster_mnt[1,x+1]
            if i==nodata:i=e                            
            dz_dx = float(f+i-(d+g))/float(4*Csize)
            dz_dy = float(g+h+i-(d+e+f))/float(3*Csize)
            expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
            if expo1<0.:
                expo[0,x]= 90.0 - expo1
            elif expo1>90.:
                expo[0,x]= 360.0 - expo1 + 90.0
            else:
                expo[0,x]= 90.0 - expo1 
        else: expo[0,x]=nodata
    # Pour derniere ligne
    x=1
    for x from 1 <= x < ncol-1:
        e = raster_mnt[nline-1,x] 
        if e > nodata:            
            d = raster_mnt[nline-1,x-1]    
            if d==nodata:d=e
            f = raster_mnt[nline-1,x+1]
            if f==nodata:f=e
            a = raster_mnt[nline-2,x-1]  
            if a==nodata:a=e
            b = raster_mnt[nline-2,x] 
            if b==nodata:b=e
            c = raster_mnt[nline-2,x+1]
            if c==nodata:c=e                            
            dz_dx = float(f+c-(d+a))/float(4*Csize)
            dz_dy = float(d+e+f-(a+b+c))/float(3*Csize)
            expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
            if expo1<0.:
                expo[nline-1,x]= 90.0 - expo1
            elif expo1>90.:
                expo[nline-1,x]= 360.0 - expo1 + 90.0
            else:
                expo[nline-1,x]= 90.0 - expo1 
        else: expo[nline-1,x]=nodata
    # Pour premiere colonne
    y=1
    for y from 1 <= y < nline-1:
        e = raster_mnt[y,0] 
        if e > nodata:            
            b = raster_mnt[y+1,0]    
            if b==nodata:b=e
            c = raster_mnt[y+1,1]
            if c==nodata:c=e
            f = raster_mnt[y,1]  
            if f==nodata:f=e
            h = raster_mnt[y+1,0] 
            if h==nodata:h=e
            i = raster_mnt[y+1,1]
            if i==nodata:i=e                            
            dz_dx = float(c+f+i-(b+e+h))/float(3*Csize)
            dz_dy = float(h+i-(b+c))/float(4*Csize)
            expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
            if expo1<0.:
                expo[y,0]= 90.0 - expo1
            elif expo1>90.:
                expo[y,0]= 360.0 - expo1 + 90.0
            else:
                expo[y,0]= 90.0 - expo1 
        else: expo[y,0]=nodata
    # Pour derniere colonne
    y=1
    for y from 1 <= y < nline-1:
        e = raster_mnt[y,ncol-1] 
        if e > nodata:            
            a = raster_mnt[y-1,ncol-2]    
            if a==nodata:a=e
            b = raster_mnt[y-1,ncol-1]
            if b==nodata:b=e
            d = raster_mnt[y,ncol-2]  
            if d==nodata:d=e
            g = raster_mnt[y+1,ncol-2] 
            if g==nodata:g=e
            h = raster_mnt[y+1,ncol-1]
            if h==nodata:h=e                            
            dz_dx = float(b+e+h-(a+d+g))/float(3*Csize)
            dz_dy = float(h+g-(b+a))/float(4*Csize)
            expo1 = 57.29578 * atan2(dz_dy, -dz_dx)
            if expo1<0.:
                expo[y,ncol-1]= 90.0 - expo1
            elif expo1>90.:
                expo[y,ncol-1]= 360.0 - expo1 + 90.0
            else:
                expo[y,ncol-1]= 90.0 - expo1 
        else: expo[y,ncol-1]=nodata
    return expo 


# Mask pour determiner l'emprise ou la matrice est non nulle
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef mask_zone(np.ndarray[dtype8_t,ndim=2] matrice):
    cdef unsigned int nline = matrice.shape[0]
    cdef unsigned int ncol = matrice.shape[1]
    cdef unsigned int top = nline
    cdef unsigned int bottom = 0
    cdef unsigned int left = ncol
    cdef unsigned int right = 0
    cdef unsigned int x = 0
    cdef unsigned int y = 0
    for y from 0 <= y < nline:
        for x from 0 <= x < ncol:
            if matrice[y,x]>0:
                if y < top: top=y
                if y > bottom: bottom=y
                if x < left: left=x
                if x > right: right=x    
    return top,bottom,left,right

# Fonction permettant de calculer la distance avec un raster de cout
@cython.boundscheck(False)
@cython.wraparound(False)
cdef calcul_distance_de_cout(np.ndarray[dtype8_t,ndim=2] from_rast,np.ndarray[dtypef32_t,ndim=2] cost_rast,
                            np.ndarray[dtype8_t,ndim=2] zone_rast,
                            double Csize,unsigned int Max_distance=100000):    
    """
    Calcule pour chaque cellule la distance de plus faible cout cumule vers la source la plus proche sur une surface de cout.
    ----------
    Parameters
    ----------
    from_rast:      ndarray int32    Raster contenant les cellules sources, un identifiant par cellule est souhaitable
    cost_rast:      ndarray float    Raster de surface de cout. Le cout est ici le cout pour traverser la cellule
    zone_rast:      ndarray int32    Raster contenant 1 sur la zone d'interet, 0 sinon
    Csize:          double           Taille de cellule pour les rasters
    Max_distance:   uint32           Distance de plus faible cout cumule maximale, par defaut 100 000 metres    
    ----------
    Returns
    ----------
    Out_distance:   ndarray int32   Raster des distances de plus faible cout cumule jusqu'a la source la plus proche
    Out_allocation: ndarray int32   Raster permettant de connaitre l'identifiant de la cellule source la plus proche
    ----------
    Examples
    --------
    >>> import ogr,gdal
    >>> import numpy as np
    >>> Out_distance,Out_allocation = calcul_distance_de_cout(Route,Ponderation_pente,Foret,5,150)
    """
    cdef unsigned int nline = from_rast.shape[0]
    cdef unsigned int ncol = from_rast.shape[1]
    cdef double diag = 1.414214*Csize
    cdef double direct = Csize
    cdef unsigned int h,b,l,r
    h,b,l,r = mask_zone(from_rast)
    # Creation des rasters de sorties
    cdef np.ndarray[dtype32_t,ndim=2] Out_distance = np.ones_like(from_rast,dtype=np.int32)*Max_distance
    cdef np.ndarray[dtype32_t,ndim=2] Out_alloc = np.ones_like(from_rast,dtype=np.int32)*-9999
    cdef unsigned int x,y,x1=l,y1=h,test,count_sans_match = 0
    cdef double Dist,dist_ac = Csize
    # Initialisation du raster
    for y1 from h <= y1 <b:
        for x1 from l <= x1 <r:
            if from_rast[y1,x1]>0:
                Out_distance[y1,x1] = 0
                Out_alloc[y1,x1] = from_rast[y1,x1]
                for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                    for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):                        
                        if zone_rast[y,x]==1:
                            if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag
                            else: Dist = cost_rast[y,x]*direct
                            if Out_distance[y,x]>Dist:
                                Out_distance[y,x] = int(Dist+0.5)                                
                                Out_alloc[y,x] = from_rast[y1,x1]
    # Traitement complet
    h,b,l,r = mask_zone(zone_rast)    
    while dist_ac<=Max_distance and count_sans_match <15*Csize:
        test = 0
        y1,x1=h,l
        for y1 from h <= y1 <b:
            for x1 from l <= x1 <r:
                if Out_distance[y1,x1]==dist_ac:
                    test=1   
                    for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                        for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                            if zone_rast[y,x]==1:
                                if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                                else: Dist = cost_rast[y,x]*direct+dist_ac
                                if Out_distance[y,x]>Dist:
                                    Out_distance[y,x] = int(Dist+0.5)
                                    Out_alloc[y,x] = Out_alloc[y1,x1]
        if test==1:count_sans_match = 0
        else:count_sans_match +=1
        dist_ac +=1
    for y in range(0,nline,1):
        for x in range(0,ncol,1):
            if Out_distance[y,x]==Max_distance:
                Out_distance[y,x]=-9999
                Out_alloc[y,x] = -9999
    return Out_distance,Out_alloc

# Fonction permettant de calculer la distance avec un raster de cout et 2 allocations
@cython.boundscheck(False)
@cython.wraparound(False)
cdef calcul_distance_de_cout_2_alloc(np.ndarray[dtype_t,ndim=2] from_rast1,np.ndarray[dtype_t,ndim=2] from_rast2,
                                    np.ndarray[dtypef_t,ndim=2] cost_rast, np.ndarray[dtype_t,ndim=2] zone_rast,
                                    double Csize,unsigned int Max_distance=100000):    
    """
    Calcule pour chaque cellule la distance de plus faible cout cumule vers la source la plus proche sur une surface de cout.
    ----------
    Parameters
    ----------
    from_rast1, 2:  ndarray int32    Raster contenant les cellules sources, un identifiant par cellule est souhaitable
    cost_rast:      ndarray float    Raster de surface de cout. Le cout est ici le cout pour traverser la cellule
    zone_rast:      ndarray int32    Raster contenant 1 sur la zone d'interet, 0 sinon
    Csize:          uint32           Taille de cellule pour les rasters
    Max_distance:   uint32           Distance de plus faible cout cumule maximale, par defaut 100 000 metres    
    ----------
    Returns
    ----------
    Out_distance:    ndarray int32   Raster des distances de plus faible cout cumule jusqu'a la source la plus proche
    Out_allocation1: ndarray int32   Raster permettant de connaitre l'identifiant 1 de la cellule source la plus proche
    Out_allocation2: ndarray int32   Raster permettant de connaitre l'identifiant 2 de la cellule source la plus proche
    """
    cdef unsigned int nline = from_rast1.shape[0]
    cdef unsigned int ncol = from_rast1.shape[1]
    cdef double diag = 1.414214*Csize
    cdef double direct = Csize
    cdef unsigned int h,b,l,r
    h,b,l,r = mask_zone(from_rast1)
    # Creation des rasters de sorties
    cdef np.ndarray[dtype_t,ndim=2] Out_distance = np.ones_like(from_rast1,dtype=np.int)*(Max_distance+1)
    cdef np.ndarray[dtype_t,ndim=2] Out_alloc1 = np.ones_like(from_rast1,dtype=np.int)*-9999
    cdef np.ndarray[dtype_t,ndim=2] Out_alloc2 = np.ones_like(from_rast1,dtype=np.int)*-9999
    cdef unsigned int x,y,x1=l,y1=h,test,count_sans_match = 0
    cdef double Dist,dist_ac = Csize
    # Initialisation du raster
    for y1 from h <= y1 <b:
        for x1 from l <= x1 <r:
            if from_rast1[y1,x1]>0:
                Out_distance[y1,x1] = 0
                Out_alloc1[y1,x1] = from_rast1[y1,x1]
                Out_alloc2[y1,x1] = from_rast2[y1,x1]
                for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                    for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):                        
                        if zone_rast[y,x]==1:
                            if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag
                            else: Dist = cost_rast[y,x]*direct
                            if Out_distance[y,x]>Dist:
                                Out_distance[y,x] = int(Dist+0.5)                                
                                Out_alloc1[y,x] = from_rast1[y1,x1]
                                Out_alloc2[y,x] = from_rast2[y1,x1]
    # Traitement complet
    h,b,l,r = mask_zone(zone_rast)    
    while dist_ac<=Max_distance and count_sans_match <15*Csize:
        test = 0
        y1,x1=h,l
        for y1 from h <= y1 <b:
            for x1 from l <= x1 <r:
                if Out_distance[y1,x1]==dist_ac:
                    test=1   
                    for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                        for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                            if zone_rast[y,x]==1:
                                if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                                else: Dist = cost_rast[y,x]*direct+dist_ac
                                if Out_distance[y,x]>Dist:
                                    Out_distance[y,x] = int(Dist+0.5)
                                    Out_alloc1[y,x] = Out_alloc1[y1,x1]
                                    Out_alloc2[y,x] = Out_alloc2[y1,x1]
        if test==1:count_sans_match = 0
        else:count_sans_match +=1
        dist_ac +=1
    for y in range(0,nline,1):
        for x in range(0,ncol,1):
            if Out_distance[y,x] > Max_distance:
                Out_distance[y,x] = -9999
                Out_alloc1[y,x] = -9999
                Out_alloc2[y,x] = -9999
    return Out_distance,Out_alloc1,Out_alloc2
   
#Renvoie la moyenne des cellules adjacentes
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtypef_t,ndim=2] focal_stat_mean(np.ndarray[dtypef_t,ndim=2] raster,double nodata,unsigned int cote):
    """
    Calcule la moyenne locale sur un raster
    -----
    Inputs: Raster_MNT, Nodata Value,nb de cellule sur le cote du masque flottant
    -----
    Output : Raster moyen
    """
    cdef unsigned int nline= raster.shape[0]
    cdef unsigned int ncol = raster.shape[1]
    cdef np.ndarray[dtypef_t,ndim=2] mean = np.ones_like(raster,dtype=np.float)*nodata
    cdef unsigned int x1=0
    cdef unsigned int y1=0
    cdef unsigned int y,x
    cdef double somme,nb
    for y1 from 0 <= y1 < nline:
        for x1 from 0 <= x1 < ncol:
            if raster[y1,x1]!=nodata:
                nb,somme,y,x=0.0,0.0,int_max(0,y1-cote),int_max(0,x1-cote)                
                # Grille sans les bordures
                for y from int_max(0,y1-cote) <= y < int_min(nline,y1+cote+1):
                    for x from int_max(0,x1-cote) <= x < int_min(ncol,x1+cote+1):
                        if raster[y,x]!=nodata:
                            somme += raster[y,x]
                            nb += 1.0
                mean[y1,x1]=somme/nb
    return mean 

#Renvoie la somme des cellules adjacentes    
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtypef_t,ndim=2] focal_stat_sum(np.ndarray[dtypef_t,ndim=2] raster,double nodata,unsigned int cote):
    """
    Calcule la moyenne locale sur un raster
    -----
    Inputs: Raster_MNT, Nodata Value,nb de cellule sur le cote du masque flottant
    -----
    Output : Raster moyen
    """
    cdef unsigned int nline= raster.shape[0]
    cdef unsigned int ncol = raster.shape[1]
    cdef np.ndarray[dtypef_t,ndim=2] rsomme = np.ones_like(raster,dtype=np.float)*nodata
    cdef unsigned int x1=0
    cdef unsigned int y1=0
    cdef unsigned int y,x
    cdef double somme
    for y1 from 0 <= y1 < nline:
        for x1 from 0 <= x1 < ncol:
            if raster[y1,x1]!=nodata:
                somme,y,x=0.0,int_max(0,y1-cote),int_max(0,x1-cote)                
                # Grille sans les bordures
                for y from int_max(0,y1-cote) <= y < int_min(nline,y1+cote+1):
                    for x from int_max(0,x1-cote) <= x < int_min(ncol,x1+cote+1):
                        if raster[y,x]!=nodata:
                            somme += raster[y,x]
                rsomme[y1,x1]=somme
    return rsomme 

#Renvoie le nombre de cellule non vides dans les cellules adjacentes
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtypef_t,ndim=2] focal_stat_nb(np.ndarray[dtypef_t,ndim=2] raster,double nodata,unsigned int cote):
    """
    Calcule le nombre de cellule non null sur un raster
    -----
    Inputs: Raster_MNT, Nodata Value,nb de cellule sur le cote du masque flottant
    -----
    Output : Raster moyen
    """
    cdef unsigned int nline= raster.shape[0]
    cdef unsigned int ncol = raster.shape[1]
    cdef np.ndarray[dtypef_t,ndim=2] rnb = np.ones_like(raster,dtype=np.float)*nodata
    cdef unsigned int x1=0
    cdef unsigned int y1=0
    cdef unsigned int y,x
    cdef double nb
    for y1 from 0 <= y1 < nline:
        for x1 from 0 <= x1 < ncol:
            if raster[y1,x1]!=nodata:
                nb,y,x=0.0,int_max(0,y1-cote),int_max(0,x1-cote)                
                # Grille sans les bordures
                for y from int_max(0,y1-cote) <= y < int_min(nline,y1+cote+1):
                    for x from int_max(0,x1-cote) <= x < int_min(ncol,x1+cote+1):
                        if raster[y,x]!=nodata:
                            nb += 1.0
                rnb[y1,x1]=nb
    return rnb 

#Renvoie le minimum des cellules adjacentes
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtypef_t,ndim=2] focal_stat_min(np.ndarray[dtypef_t,ndim=2] raster,double nodata,unsigned int cote):
    """
    Calcule le minimum local sur un raster
    -----
    Inputs: Raster_MNT, Nodata Value,nb de cellule sur le cote du masque flottant
    -----
    Output : Raster min
    """
    cdef unsigned int nline= raster.shape[0]
    cdef unsigned int ncol = raster.shape[1]
    cdef np.ndarray[dtypef_t,ndim=2] rmin = np.ones_like(raster,dtype=np.float)*nodata
    cdef unsigned int x1=0
    cdef unsigned int y1=0
    cdef unsigned int y,x
    cdef double max_value = np.max(raster[raster!=nodata])
    cdef double local_min = max_value
    for y1 from 0 <= y1 < nline:
        for x1 from 0 <= x1 < ncol:
            if raster[y1,x1]!=nodata:
                local_min,y,x=max_value,int_max(0,y1-cote),int_max(0,x1-cote)                
                # Grille sans les bordures
                for y from int_max(0,y1-cote) <= y < int_min(nline,y1+cote+1):
                    for x from int_max(0,x1-cote) <= x < int_min(ncol,x1+cote+1):
                        if raster[y,x]!=nodata:
                            local_min = double_min(local_min,raster[y,x])
                rmin[y1,x1]=local_min
    return rmin 

#Renvoie le maximum des cellules adjacentes
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtypef_t,ndim=2] focal_stat_max(np.ndarray[dtypef_t,ndim=2] raster,double nodata,unsigned int cote):
    """
    Calcule le maximum local sur un raster
    -----
    Inputs: Raster_MNT, Nodata Value,nb de cellule sur le cote du masque flottant
    -----
    Output : Raster max
    """
    cdef unsigned int nline= raster.shape[0]
    cdef unsigned int ncol = raster.shape[1]
    cdef np.ndarray[dtypef_t,ndim=2] rmax = np.ones_like(raster,dtype=np.float)*nodata
    cdef unsigned int x1=0
    cdef unsigned int y1=0
    cdef unsigned int y,x
    cdef double min_value = np.min(raster[raster!=nodata])
    cdef double local_max = min_value
    for y1 from 0 <= y1 < nline:
        for x1 from 0 <= x1 < ncol:
            if raster[y1,x1]!=nodata:
                local_max,y,x=min_value,int_max(0,y1-cote),int_max(0,x1-cote)                
                # Grille sans les bordures
                for y from int_max(0,y1-cote) <= y < int_min(nline,y1+cote+1):
                    for x from int_max(0,x1-cote) <= x < int_min(ncol,x1+cote+1):
                        if raster[y,x]!=nodata:
                            local_max = double_max(local_max,raster[y,x])
                rmax[y1,x1]=local_max
    return rmax 

##############################################################################################################################################
### Fonctions pour le modele cable
##############################################################################################################################################

# Renvoie la zone correspondant a l'azimuth cherche
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype8_t,ndim=2] get_zone(int direction,np.ndarray[dtype8_t,ndim=2] Matrice,int Buffer_cote):
    cdef unsigned int h3 = direction*(Buffer_cote+1)
    cdef unsigned int b3 = h3 + Buffer_cote+1
    return Matrice[h3:b3]

# Renvoie le cadran correspondant a l'azimuth
@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_cadran(unsigned int direction,unsigned int Buffer_cote):
    cdef unsigned int h,l
    if direction <=90:
        h = 0        
        l = Buffer_cote
    elif direction <=180:
        h = Buffer_cote
        l = Buffer_cote
    elif direction <=270:
        h = Buffer_cote
        l = 0
    else:
        h = 0
        l = 0
    return h,h+Buffer_cote+1,l,l+Buffer_cote+1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_npix(int az, int npix,int coordY, int coordX, int ncols, int nrows,
               np.ndarray[dtype16_t,ndim=2] Row_line, np.ndarray[dtype16_t,ndim=2] Col_line):                   
    cdef int i=0,x,y
    for i from 0<=i<npix:
        x = Col_line[az,i]+coordX
        if x<0:
            break
        if x>=ncols:
            break       
        y = Row_line[az,i]+coordY
        if y<0:
            break
        if y>=nrows:
            break        
    return i      


#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef check_line(np.ndarray[dtypef_t,ndim=2] Line,int Lmax,int Lmin,int nrows, int ncols,double Lsans_foret):
#    cdef int indmax = 0
#    cdef int npix = Line.shape[0]
#    cdef int test = 1
#    cdef int i=0
#    cdef int testdist
#    cdef double Lline=Lmin-1,Dsansforet=0.
#    for i from 0<=i<npix: 
#        if Line[i,5]<0:break
#        if Line[i,5]>=ncols:break
#        if Line[i,6]<0:break
#        if Line[i,6]>=nrows:break
#        if Line[i,7]==1:break
#        if sqrt(Line[i,0]*Line[i,0]+(Line[i,1]-Line[0,1])*(Line[i,1]-Line[0,1]))>Lmax:break    
#        if Line[i,2]==1:
#            indmax = i 
#            Dsansforet=0
#        else:
#            if i>0: Dsansforet+=Line[i,0]-Line[i-1,0]
#            if Dsansforet>=Lsans_foret:break        
#    Lline = Line[indmax,0]
#    if Lline <= Lmin:
#        test=0
#    return test,indmax+1,Lline

@cython.boundscheck(False)
@cython.wraparound(False)
cdef check_line(np.ndarray[dtypef_t,ndim=2] Line,int Lmax,int Lmin,int nrows, int ncols,double Lsans_foret,double Lslope,double PropSlope):
    cdef int indmax = 0,indmax2=0
    cdef int npix = Line.shape[0]
    cdef int test = 1
    cdef int i=0
    cdef int testdist
    cdef double Lline=Lmin-1,Dsansforet=0.,Dcum=0.,D=0
    for i from 0<=i<npix: 
        if Line[i,5]<0:break
        if Line[i,5]>=ncols:break
        if Line[i,6]<0:break
        if Line[i,6]>=nrows:break
        if Line[i,7]==1:break
        if sqrt(Line[i,0]*Line[i,0]+(Line[i,1]-Line[0,1])*(Line[i,1]-Line[0,1]))>Lmax:break    
        ########    Devers        
        if i>0:
            if Line[i,8]+Line[i,9]==0:
                Dcum += Line[i,0]-Line[i-1,0]
            if Dcum>Lslope:break            
            if Dcum/Line[i,0]<PropSlope:                
                indmax2=i
                Line[i,9]=1
            else:
                Line[i,9]=0
        else:
            Line[i,9]=1
        Line[i,8]=Dcum
        ########    Longueur sans foret
        if Line[i,2]==1:
            indmax = i 
            Dsansforet=0
        else:
            if i>0: Dsansforet+=Line[i,0]-Line[i-1,0]
            if Dsansforet>=Lsans_foret:break   
    indmax =  int_min(indmax,indmax2)   
    Lline = Line[indmax,0]
    if Lline <= Lmin:
        test=0
    return test,indmax+1,Lline


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cdef get_line_carac_simple(int coordX, int coordY, int az,double Csize,int ncols,int nrows,
                            double Lline,np.ndarray[dtype16_t,ndim=2] Row_ext,np.ndarray[dtype16_t,ndim=2] Col_ext,
                            np.ndarray[dtypef_t,ndim=2] D_ext, np.ndarray[dtype8_t,ndim=2] Forest, np.ndarray[dtype8_t,ndim=2] Rast_couv):
    cdef int x,y                          
    cdef int i=0
    cdef int nfor=0
    cdef double Dmoy_car=0,Forest_area=0
    while D_ext[az,i]<=Lline:
        x = Col_ext[az,i]+coordX
        if x<0:
            i+=1
            continue
        if x>=ncols:
            i+=1
            continue          
        y=Row_ext[az,i]+coordY 
        if y<0:
            i+=1
            continue
        if y>=nrows:
            i+=1
            continue
        if Forest[y,x]==1:
            nfor+=1
            Dmoy_car+=D_ext[az,i]
            Rast_couv[y,x]=1
        i+=1
    if nfor>0:
        Forest_area=nfor*Csize*Csize
        Dmoy_car = Dmoy_car/nfor       
    else:
        Dmoy_car = D_ext[az,i]
    return int(Dmoy_car),int(Forest_area),Rast_couv

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cdef get_line_carac_vol(int coordX, int coordY, int az,double Csize,int ncols,int nrows,
                         double Lline,np.ndarray[dtype16_t,ndim=2] Row_ext,np.ndarray[dtype16_t,ndim=2] Col_ext,
                         np.ndarray[dtypef_t,ndim=2] D_ext, np.ndarray[dtype8_t,ndim=2] Forest, np.ndarray[dtype8_t,ndim=2] Rast_couv,
                         np.ndarray[dtypef_t,ndim=2] Vol_ha, np.ndarray[dtypef_t,ndim=2] Vol_AM):
    cdef int x,y                          
    cdef int i=0
    cdef int nfor=0,nvam=0
    cdef double Dmoy_car=0,Forest_area=0
    cdef double Dmoy_car2=0,Vtot=0,VAM=0
    while D_ext[az,i]<=Lline:
        x = Col_ext[az,i]+coordX
        if x<0:
            i+=1
            continue
        if x>=ncols:
            i+=1
            continue          
        y=Row_ext[az,i]+coordY 
        if y<0:
            i+=1
            continue
        if y>=nrows:
            i+=1
            continue
        if Forest[y,x]==1:
            nfor+=1
            Dmoy_car2+=D_ext[az,i]
            Rast_couv[y,x]=1
        if Vol_ha[y,x]>0:
            Vtot+=Vol_ha[y,x]*Csize*Csize*0.0001
            Dmoy_car+=Vol_ha[y,x]*D_ext[az,i]*Csize*Csize*0.0001
        if Vol_AM[y,x]>0:
            nvam +=1
            VAM+=Vol_AM[y,x]
        i+=1
    Forest_area=nfor*Csize*Csize
    if Vtot>0:
        Dmoy_car = Dmoy_car/Vtot
    else:
        Vtot=-1
        if nfor>0:
            Dmoy_car = Dmoy_car2/nfor
        else:
            Dmoy_car = D_ext[az,i]
    if nvam>0:
        VAM = VAM/nvam
    else:
        VAM = -1      
    return int(Dmoy_car),int(Forest_area),int(Vtot),int(10*VAM),Rast_couv

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Check_line(int coordX, int coordY, int az,int ncols,int nrows, double Lline,
                 np.ndarray[dtype16_t,ndim=2] Row_ext,np.ndarray[dtype16_t,ndim=2] Col_ext,
                 np.ndarray[dtypef_t,ndim=2] D_ext, np.ndarray[dtypef_t,ndim=2] D_lat,
                 np.ndarray[dtype8_t,ndim=2] Rast_couv,double debut, double recouv,double rapport):
    cdef int x,y                          
    cdef int i=0,test=1
    cdef int nb1=0,nb2=0
    cdef np.ndarray[dtype8_t,ndim=2] Rast_couv_bis = np.zeros((nrows,ncols),dtype=np.int8)  
    while D_ext[az,i]<=Lline:
        x = Col_ext[az,i]+coordX
        if x<0:
            i+=1
            continue
        if x>=ncols:
            i+=1
            continue          
        y=Row_ext[az,i]+coordY 
        if y<0:
            i+=1
            continue
        if y>=nrows:
            i+=1
            continue        
        if Rast_couv[y,x]==0:                           
            if D_ext[az,i]<debut:
                Rast_couv_bis[y,x]=2
                nb2 +=1
            else:
                if D_lat[az,i]<recouv:
                    Rast_couv_bis[y,x]=1
                    nb1 +=1
                else:
                    Rast_couv_bis[y,x]=2
                    nb2 +=1
        elif Rast_couv[y,x]==1:  
            test=0
            break
        else:
            nb2 +=1
        i+=1
    #Check if the line is really important 
    if test and nb1 >= (nb1+nb2)*rapport:
        Rast_couv = Rast_couv+Rast_couv_bis
    else:
        test=0
    return test,Rast_couv

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Check_line2(int coordX, int coordY, int az,int ncols,int nrows, double Lline,
                 np.ndarray[dtype16_t,ndim=2] Row_ext,np.ndarray[dtype16_t,ndim=2] Col_ext,
                 np.ndarray[dtypef_t,ndim=2] D_ext, np.ndarray[dtypef_t,ndim=2] D_lat,
                 np.ndarray[dtype8_t,ndim=2] Rast_couv, double recouv,double rapport):
    cdef int x,y                          
    cdef int i=0,test=1
    cdef int nb1=0,nb2=0
    cdef np.ndarray[dtype8_t,ndim=2] Rast_couv_bis = np.zeros((nrows,ncols),dtype=np.int8)  
    cdef double  debut = double_min(100,Lline*0.4)
    while D_ext[az,i]<=Lline:
        x = Col_ext[az,i]+coordX
        if x<0:
            i+=1
            continue
        if x>=ncols:
            i+=1
            continue          
        y=Row_ext[az,i]+coordY 
        if y<0:
            i+=1
            continue
        if y>=nrows:
            i+=1
            continue        
        Rast_couv_bis[y,x]=1
        if Rast_couv[y,x]==0:            
            nb1+=1
        elif Rast_couv[y,x]>1 and D_ext[az,i]>debut and D_lat[az,i]<recouv:
            test=0
            break
        else:
            nb2+=1            
        i+=1
    #Check if the line is really important 
    if test and nb1 >= (nb1+nb2)*rapport:
        Rast_couv = Rast_couv+Rast_couv_bis
    else:
        test=0
    return test,Rast_couv

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef get_prop(int coordX, int coordY, int az,int ncols,int nrows, double Lline,
                 np.ndarray[dtype16_t,ndim=2] Row_ext,np.ndarray[dtype16_t,ndim=2] Col_ext,
                 np.ndarray[dtypef_t,ndim=2] D_ext, np.ndarray[dtypef_t,ndim=2] D_lat,
                 np.ndarray[dtype8_t,ndim=2] Rast_couv):
    cdef int x,y                          
    cdef int i=0,test=1
    cdef double nb1=0,nb2=0
    while D_ext[az,i]<=Lline:
        x = Col_ext[az,i]+coordX
        if x<0:
            i+=1
            continue
        if x>=ncols:
            i+=1
            continue          
        y=Row_ext[az,i]+coordY 
        if y<0:
            i+=1
            continue
        if y>=nrows:
            i+=1
            continue  
        if Rast_couv[y,x]==1:            
            nb1+=1
        elif Rast_couv[y,x]>1 :
            nb2+=1 
        i+=1
    return nb1/(nb1+nb2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Check_line3(int coordX, int coordY, int az,int ncols,int nrows, double Lline,
                 np.ndarray[dtype16_t,ndim=2] Row_ext,np.ndarray[dtype16_t,ndim=2] Col_ext,
                 np.ndarray[dtypef_t,ndim=2] D_ext, np.ndarray[dtypef_t,ndim=2] D_lat,
                 np.ndarray[dtype8_t,ndim=2] Rast_couv, double rapport):
    cdef int x,y                          
    cdef int i=0,test=1
    cdef int nb1=0,nb2=0
    cdef np.ndarray[dtype8_t,ndim=2] Rast_couv_bis = np.copy(Rast_couv)  
    cdef double  debut = double_min(100,Lline*0.4)
    while D_ext[az,i]<=Lline:
        x = Col_ext[az,i]+coordX
        if x<0:
            i+=1
            continue
        if x>=ncols:
            i+=1
            continue          
        y=Row_ext[az,i]+coordY 
        if y<0:
            i+=1
            continue
        if y>=nrows:
            i+=1
            continue        
        if Rast_couv[y,x]>0:
            Rast_couv_bis[y,x]-=1
        if Rast_couv[y,x]==1:            
            nb1+=1
        elif Rast_couv[y,x]>1 :
            if D_ext[az,i]>debut :
                nb2+=1 
            else:
                nb1+=1        
        i+=1
    #Check if the line is really important 
    if nb1 < (nb1+nb2)*rapport:
        test=0
        Rast_couv = Rast_couv_bis
    return test,Rast_couv





#####################################################
# Fonction pour equation cable
#####################################################

# Update Fcariage according to Lo
cdef double Fcariage(double Lo,double F,double q2,double q3,double s1, double Dsupdep=0.,double Dsupend=0.):
    return (0.5*((s1+Dsupdep)*q2+((Lo-s1)+Dsupend)*q3))*g + F

# Derivee de fx selon Th
@cython.cdivision(True)
cdef double df_dTh(double Th,double Tv,double F,double W,double s1,double Lo,
                   double EAo,double rac1,double rac2,double rac3,double rac4):
    cdef double a = 1./Th*(-Tv/rac1 + (Tv-F-W)/rac2-(Tv-F-W*s1/Lo)/rac3 + (Tv-W*s1/Lo)/rac4)                  
    a += asinh(Tv/Th)-asinh((Tv-F-W)/Th)+asinh((Tv-F-W*s1/Lo)/Th)-asinh((Tv-W*s1/Lo)/Th)
    a *= Lo/W
    a += Lo/EAo
    return a

# Derivee de fz selon Th
@cython.cdivision(True) 
cdef double dg_dTh(double Tv,double Th,double Lo,double W,double s1,double F,
                   double rac1,double rac2,double rac3,double rac4):
    cdef double a = 1./(Th*Th)*(-Tv*Tv/rac1+(Tv-F-W)*(Tv-F-W)/rac2-(Tv-F-W*s1/Lo)*(Tv-F-W*s1/Lo)/rac3+(Tv-W*s1/Lo)*(Tv-W*s1/Lo)/rac4)
    a += sqrt(1+(Tv/Th)*(Tv/Th))-sqrt(1+((Tv-F-W)/Th)*((Tv-F-W)/Th))+sqrt(1+((Tv-F-W*s1/Lo)/Th)*((Tv-F-W*s1/Lo)/Th))-sqrt(1+((Tv-W*s1/Lo)/Th)*((Tv-W*s1/Lo)/Th))
    a *= Lo/W
    return a

# Equation F(Th,Tv)
@cython.cdivision(True) 
cdef double f_x(double Th,double Tv,double Lo,double EAo,double W,double F,double s1,double D):
    cdef double x = Th*Lo/EAo -D
    x += Th*Lo/W*(asinh(Tv/Th)-asinh((Tv-F-W)/Th)+asinh((Tv-F-W*s1/Lo)/Th)-asinh((Tv-W*s1/Lo)/Th))
    return x

# Equation G(Th,Tv)
@cython.cdivision(True)     
cdef double f_z(double Th,double Tv,double Lo,double EAo,double W,double F,double s1,double H):
    cdef double z = W*Lo/EAo*(Tv/W-0.5)-H
    cdef double Temp= sqrt(1+(Tv/Th)*(Tv/Th))-sqrt(1+((Tv-F-W)/Th)*((Tv-F-W)/Th))+F*W/(Th*EAo)*(s1/Lo-1)
    Temp += sqrt(1+((Tv-F-W*s1/Lo)/Th)*((Tv-F-W*s1/Lo)/Th))-sqrt(1+((Tv-W*s1/Lo)/Th)*((Tv-W*s1/Lo)/Th))
    z += Th*Lo/W*Temp
    return z

# Fonction donnant X en fonction de s, l'abscisse curviligne
@cython.cdivision(True)
cdef double calcul_xs(double Th,double Tv,double Lo,double EAo,double W,double F,double s1,double s):
    """    
    Renvoie la position horizontale du cable en s quand la charge se trouve a l'abscisse curviligne s1
    ---
    Input:
        Th    N       Composante horizontale de la tension au support le plus haut quand la charge est en s1. Estimation
        Tv    N       Composante verticale de la tension au support le plus haut quand la charge est en s1. Estimation        
        Lo    m       Longueur a vide et sans elasticite du cable porteur sur la section
        EAo   N       Module de Young multiplie par la section du cable porteur
        W     N       Poids exerce par la masse du cable sur toute sa longueur
        F     N       Force de gravite exercee par la charge
        s1    m       Abscisse curviligne correspondant a la position de la charge
        s    m        Abscisse curviligne correspondant a la position du cable testee
    """    
    cdef double x = Th*s/EAo
    x += Th*Lo/W*(asinh(Tv/Th)-asinh((Tv-F-W*s/Lo)/Th)+asinh((Tv-F-W*s1/Lo)/Th)-asinh((Tv-W*s1/Lo)/Th))
    return x
    
# Fonction donnant Z en fonction de s, l'abscisse curviligne    
@cython.cdivision(True)
cdef double calcul_zs(double Th,double Tv,double Lo,double EAo,double W,double F,double s1,double s):
    """    
    Renvoie la position verticale du cable en s quand la charge se trouve a l'abscisse curviligne s1
    ---
    Input:
        Th    N       Composante horizontale de la tension au support le plus haut quand la charge est en s1. Estimation
        Tv    N       Composante verticale de la tension au support le plus haut quand la charge est en s1. Estimation        
        Lo    m       Longueur a vide et sans elasticite du cable porteur sur la section
        EAo   N       Module de Young multiplie par la section du cable porteur
        W     N       Poids exerce par la masse du cable sur toute sa longueur
        F     N       Force de gravite exercee par la charge
        s1    m       Abscisse curviligne correspondant a la position de la charge
        s    m        Abscisse curviligne correspondant a la position du cable testee
    """   
    cdef double z = W*s/EAo*(Tv/W-s/(2*Lo))
    cdef double Temp = sqrt(1+square(Tv/Th))-sqrt(1+square((Tv-F-W*s/Lo)/Th))+F*W/(Th*EAo)*(s1/Lo-s/Lo)
    Temp += sqrt(1+square((Tv-F-W*s1/Lo)/Th))-sqrt(1+square((Tv-W*s1/Lo)/Th))
    z += Th*Lo/W*Temp
    return z
   
# FindThTv according to Tmax
cdef tuple find_ThTvTmax(double Tmax,double W,double EAo,double F,double pas,double D,double H,double Lo,unsigned int step=50):
    cdef double Fx,Gz,Fx_min=0.05,Gz_min=0.05,sum_min=10.
    cdef double Th=-1.,Tv=-1.
    cdef Py_ssize_t i, j,test=0
    cdef unsigned int T=1
    for i from 0 <= i < int(Tmax) by step:
        for j from 0 <= j < int(Tmax) by step:
            Fx = f_x(float(i),float(j),Lo,EAo,W,F,pas,D)            
            if fabs(Fx)< Fx_min:
                Gz = f_z(float(i),float(j),Lo,EAo,W,F,pas,H)
                if fabs(Gz)<Gz_min:                    
                    if fabs(Fx)+fabs(Gz)<sum_min:
                        sum_min = double_min(fabs(Fx)+fabs(Gz),sum_min)
                        Th = float(i)
                        Tv = float(j)
                        if sum_min<0.03:
                            test=1
                            break
        if test==1:break
    if sqrt(Th*Th+Tv*Tv)>Tmax: T=0
    return Th,Tv,T

# Trouve le couple Th,Tv solution des equations fx = 0 et fz=0
@cython.cdivision(True)
cdef tuple newton_ThTv(double Th,double Tv,double H,double D,double Lo,double W,double s1,double F,double EAo,double Tmax,double err=1.0):
    """
    Cherche Th et Tv pour resoudre les equations de la position de la charge selon l'abscisse curviligne
    Renvoie le couple Th,Tv solution des equations FX(Th,Tv)=0 et GZ(Th,Tv)=0
    ---
    Input:
        Th    N       Composante horizontale de la tension au support le plus haut quand la charge est en s1. Estimation
        Tv    N       Composante verticale de la tension au support le plus haut quand la charge est en s1. Estimation
        H     m       Difference d'altitude entre les deux supports
        D     m       Distance horizontale entre les deux supports
        Lo    m       Longueur a vide et sans elasticite du cable porteur sur la section
        W     N       Poids exerce par la masse du cable sur toute sa longueur
        s1    m       Abscisse curviligne correspondant a la position de la charge
        F     N       Force de gravite exercee par la charge
        EAo   N       Module de Young multiplie par la section du cable porteur
        err           Precision recherchee pour Th,Tv. Par defaut 1.0 N
    """    
    cdef double h = 100.0
    cdef double k = 100.0
    cdef unsigned int it = 0,step=100
    cdef double fo,go,rac1,rac2,rac3,rac4,dfTv,dfTh,dgTv,dgTh,coeff
    cdef double Fx,Gz,Fx_min=0.05,Gz_min=0.05,sum_min=10.
    cdef unsigned int i, j,test=0
    while fabs(h)>err and fabs(k)>err :
        fo = f_x(Th,Tv,Lo,EAo,W,F,s1,D) #fo
        go = f_z(Th,Tv,Lo,EAo,W,F,s1,H) #go
        rac1 = sqrt((Tv/Th)*(Tv/Th)+1)
        rac2 = sqrt(((Tv-F-W)/Th)*((Tv-F-W)/Th)+1)
        rac3 = sqrt(((Tv-F-W*s1/Lo)/Th)*((Tv-F-W*s1/Lo)/Th)+1)
        rac4 = sqrt(((Tv-W*s1/Lo)/Th)*((Tv-W*s1/Lo)/Th)+1)
        dfTv = Lo/W*(1.0/rac1-1.0/rac2+1.0/rac3-1.0/rac4)
        dfTh = df_dTh(Th,Tv,F,W,s1,Lo,EAo,rac1,rac2,rac3,rac4)
        dgTv = Lo/EAo + Lo/(W*Th)*(Tv/rac1-(Tv-F-W)/rac2+(Tv-F-W*s1/Lo)/rac3-(Tv-W*s1/Lo)/rac4)       
        dgTh = dg_dTh(Tv,Th,Lo,W,s1,F,rac1,rac2,rac3,rac4)
        coeff = 1.0/(dfTh*dgTv-dfTv*dgTh)        
        h = coeff*(-dgTv*fo+dfTv*go)
        k = coeff*(dgTh*fo-dfTh*go)
        Th = Th + h
        Tv = Tv + k
        it += 1
        if double_min(Th,Tv) < 0:      
            it = 0
            for i from 0 <= i < ceil(Tmax) by step:
                for j from 0 <= j < ceil(Tmax) by step:
                    Fx = f_x(float(i),float(j),Lo,EAo,W,F,s1,D)            
                    if fabs(Fx)< Fx_min:
                        Gz = f_z(float(i),float(j),Lo,EAo,W,F,s1,H)
                        if fabs(Gz)<Gz_min:                    
                            if fabs(Fx)+fabs(Gz)<sum_min:
                                sum_min = double_min(fabs(Fx)+fabs(Gz),sum_min)
                                Th = float(i)
                                Tv = float(j)
                                if sum_min<0.03:
                                    test=1
                                    break
                if test:break                        
        if it>20: break
    return (Th,Tv) 

# Generate Tab to find fastly Th,Tv,Losup
@cython.cdivision(True) 
@cython.boundscheck(False)
@cython.wraparound(False)    
cdef Tabmesh(double d,double E,double Tmax,double Lmax,double Fo,double q1,double q2,double q3,double Csize):
    """
    Creer les rasters pour converger plus vite vers les solutions Th,Tv,Lomin
    """
#    cdef unsigned int pas = int_max(ceil(Csize),5)
    cdef int pas = 1
    cdef Py_ssize_t  ncol = ceil((Lmax+Csize)/pas-1)
    cdef Py_ssize_t  nline = ncol+1
    cdef np.ndarray[dtypef_t,ndim=2] rastLosup = np.zeros((nline,ncol),dtype=np.float)
    cdef Py_ssize_t  i,j
    cdef double NaN = np.nan
    for i from 0<=i<nline:
        for j from 0<=j<ncol:
            rastLosup[i,j]=NaN
    cdef np.ndarray[dtypef_t,ndim=2] rastTh = np.copy(rastLosup)
    cdef np.ndarray[dtypef_t,ndim=2] rastTv = np.copy(rastLosup)
    cdef Py_ssize_t col=0,lig=0,Hmax  
    cdef double Tvo = 0.1*Tmax,Tho = 0.9*Tmax, Lsupo = 0.
    cdef double D,H,Tvprec,Thprec,Lsup_prec,diag,Lo,W,F,Th,Tv,Tcalc,incr,signe
    cdef double EAo = 0.25*pi*(d*d)*E
    for D from 5 <=D < ceil(Lmax+Csize) by pas:
        lig=0     
        Hmax = ceil(sqrt(Lmax*Lmax-D*D)+Csize)
        Tvprec = Tvo
        Thprec = Tho
        Lsup_prec = Lsupo
        for H from 0 <= H < Hmax by pas:
            diag =sqrt(H**2+D**2)
            Lo = diag+Lsup_prec
            W = q1*g*Lo
            F =  Fcariage(Lo,Fo,q2,q3,0.,0.)
            Th,Tv = newton_ThTv(Thprec,Tvprec,H,D,Lo,W,Lo*0.5,F,EAo,Tmax*2,0.1)
            # Check if Th,Tv are OK
            if fabs(f_x(Th,Tv,Lo,EAo,W,F,Lo*0.5,D))+fabs(f_z(Th,Tv,Lo,EAo,W,F,Lo*0.5,H))>0.01:
                Th,Tv = newton_ThTv(D/diag*Tmax,Tmax*(H/diag+0.01),H,D,Lo,W,Lo*0.5,F,EAo,Tmax*2)
                if fabs(f_x(Th,Tv,Lo,EAo,W,F,Lo*0.5,D))+fabs(f_z(Th,Tv,Lo,EAo,W,F,Lo*0.5,H))>0.01:
                    Th,Tv,T = find_ThTvTmax(Tmax,W,EAo,F,Lo*0.5,D,H,Lo,20)
                    Th,Tv = newton_ThTv(Th,Tv,H,D,Lo,W,Lo*0.5,F,EAo,Tmax*2)
                    if fabs(f_x(Th,Tv,Lo,EAo,W,F,Lo*0.5,D))+fabs(f_z(Th,Tv,Lo,EAo,W,F,Lo*0.5,H))>0.01:
                        continue                          
            # Find Lo so that Tcalc does not exceed Tmax
            Tcalc = sqrt(Th*Th+Tv*Tv)
            if ceil(Lsup_prec*0.1)>=1:
                incr = 1.
            elif ceil(Lsup_prec)>=1:
                incr = 0.1
            elif ceil(Lsup_prec*10.)>=1:
                incr = 0.01
            else:
                incr=0.001
            signe = (Tcalc-Tmax)/fabs(Tcalc-Tmax)
            # Incrementation            
            while fabs(Tcalc-Tmax) > 10.:
                Lo += signe*incr
                F =  Fcariage(Lo,Fo,q2,q3,0.,0.)
                W = q1*g*Lo
                Th,Tv = newton_ThTv(Th,Tv,H,D,Lo,W,Lo*0.5,F,EAo,Tmax*2)   
                Tcalc = sqrt(Th*Th+Tv*Tv)
                if signe*(Tcalc-Tmax)<0:                
                    incr *= 0.1
                    signe *=-1.                
                if Lo>sqrt(H**2+D**2)+1000.:
                    break
            if fabs(f_x(Th,Tv,Lo,EAo,W,F,Lo*0.5,D))+fabs(f_z(Th,Tv,Lo,EAo,W,F,Lo*0.5,H))>0.01:
                continue
            Tvprec = Tv
            Thprec = Th
            Lsup_prec = Lo-diag
            rastLosup[lig,col]=Lsup_prec
            rastTh[lig,col] = Th
            rastTv[lig,col] = Tv
            if H==0:
                Tvo = Tv
                Tho = Th
                Lsupo = Lsup_prec
            lig+=1
        col+=1
    return rastLosup,rastTh,rastTv    

@cython.cdivision(True)    
cdef double frottement(double Tension,double coeff_frot,double tan_avant,double tan_apres):
    cdef double alpha = atan(tan_avant)
    cdef double beta = atan(tan_apres)
    cdef double gama = (beta+alpha)*0.5
    cdef double num = tan(coeff_frot)*sin(gama-beta)+cos(beta-gama)
    cdef double denum = tan(coeff_frot)*sin(gama-alpha)+cos(alpha-gama)
    return Tension*num/denum

@cython.cdivision(True) 
cdef double frottement_inv(double Tension,double coeff_frot,double tan_avant,double tan_apres):
    cdef double alpha = atan(tan_avant)
    cdef double beta = atan(tan_apres)
    cdef double gama = (beta+alpha)*0.5
    cdef double num = tan(coeff_frot)*sin(gama-beta)+cos(beta-gama)
    cdef double denum = tan(coeff_frot)*sin(gama-alpha)+cos(alpha-gama)
    return Tension*denum/num

@cython.cdivision(True) 
cdef double mainline(double F,double tan_mainline,double tan_avant,double tan_apres,double sens=1):
    cdef double alpha = atan(fabs(tan_avant))
    cdef double beta = atan(fabs(tan_apres))
    cdef double lambdas = atan(fabs(tan_mainline))    
    cdef double Tchar = F*cos(lambdas)/(sin(alpha-lambdas)+sin(lambdas+sens*beta))
    return fabs(Tchar)

@cython.cdivision(True) 
cdef mainline2(double F,double tan_mainline,double tan_avant,double tan_apres,double sens=1):
    cdef double alpha = atan(fabs(tan_avant))
    cdef double beta = atan(fabs(tan_apres))
    cdef double lambdas = atan(fabs(tan_mainline))    
    cdef double Tmain = F*(cos(beta)-cos(alpha))/sin(alpha+sens*beta)
    cdef double Tchar = F*cos(lambdas)/(sin(alpha-lambdas)+sin(lambdas+sens*beta))
    return fabs(Tchar),fabs(Tmain)

@cython.cdivision(True) 
cdef double frottement_av(double Tension,double coeff_frot,double tan_haut,double tan_bas):   
    cdef double a = atan(tan_haut),T,g,tanphi,num,denum
    cdef double b = atan(tan_bas)    
    if b<a or coeff_frot==0:
        T = Tension
    else:
        g = (a+b)*0.5
        tanphi = tan(coeff_frot)
        num = tanphi*sin(g)*cos(b)+cos(g)*cos(b)-sin(b)*tanphi*cos(g)+sin(b)*sin(g)
        denum = -tanphi*cos(g)*sin(a)+sin(g)*sin(a)+cos(a)*tanphi*sin(g)+cos(a)*cos(g) 
        T = Tension*num/denum
    return T
    
@cython.cdivision(True)     
cdef double frottement_ap(double Tension,double coeff_frot,double tan_haut,double tan_bas,double charge_posi): 
    cdef double T,a,b,g,tanphi,num,denum
    if coeff_frot==0:
        T = Tension
    else:
        a = atan(tan_bas)
        b = atan(tan_haut)
        if charge_posi>=0:
            if a<b :
                T = Tension
            else:
                g = (a+b)*0.5
                tanphi = tan(coeff_frot)
                num = tanphi*sin(b-g)+cos(b-g)
                denum = tanphi*sin(a-g)+cos(a-g) 
                T = Tension*num/denum
        else:
            g = (a+b)*0.5
            tanphi = tan(coeff_frot)
            if a<b:
                num = tanphi*sin(g-b)+cos(b-g)
                denum = tanphi*sin(a+g)+cos(a+g) 
                T = Tension*num/denum
            else:
                num = tanphi*sin(g-b)-cos(b+g)
                denum = tanphi*sin(a-g)+cos(a-g) 
                T = Tension*num/denum
    return T

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int check_droite(double fact,double H,double D,double Xup,double Zup,
                       np.ndarray[dtypef_t,ndim=2] Line,double Hline_min,
                       double Hline_max,double Tmax,double q1, double q2,double q3,
                       double Fo,int pg,int pd,double Dsupdep=0.,double Dsupend=0.):
    cdef int test=1,i
    cdef double droite
    cdef double L=sqrt(H*H+D*D)
    cdef double F = (0.5*((0.5*L+Dsupdep)*q2+(0.5*L+Dsupend)*q3))*g + Fo 
    cdef double fleche = 1.1*(F*L/(4*Tmax)+q1*9.80665*L*L/(8*Tmax))
    for i in range(pg+1,pd):
        droite = -fact*H/D*(Line[i,0]-Xup)+Zup-Line[i,1]
        if droite < Hline_min:
            test=0
            break
        if droite-fleche > Hline_max:
            test=0
            break
    return test  

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double H_mid(double Lo,double F,double Th,double Tv,double Xup,double Zup,double fact,np.ndarray[dtypef_t,ndim=1] Alts,
                  double Hline_min,double q1,double EAo):
    cdef double W = q1*g*Lo
    cdef double s1 = Lo*0.5
    cdef double xcoord = Xup+fact*calcul_xs(Th,Tv,Lo,EAo,W,F,s1,s1)
    cdef double zcoord = Zup-calcul_zs(Th,Tv,Lo,EAo,W,F,s1,s1)
    cdef Py_ssize_t ind = int(xcoord*2)
    return  zcoord-(Alts[ind]+Hline_min)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double slope_H_mid(double Lo,double F,double Th,double Tv,double Xup,double Zup,double fact,np.ndarray[dtypef_t,ndim=1] Alts,
                  double q1,double EAo):
    cdef double W = q1*g*Lo
    cdef double s1 = Lo*0.5
    cdef double xcoord = Xup+fact*calcul_xs(Th,Tv,Lo,EAo,W,F,s1,s1)
    cdef double zcoord = Zup-calcul_zs(Th,Tv,Lo,EAo,W,F,s1,s1)
    return  atan((Zup-zcoord)/abs(Xup-xcoord))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double check_Hlinemin(np.ndarray[dtypef_t,ndim=1] Alts,double H,double D,double Lo,double fact,double Tho,double Tvo,
                     double Xup,double Zup,double Fo,double Tmax,double Hline_min,double Hline_max,double q1,double q2,
                     double q3,double Csize,double EAo,double Dsupdep=0.,double Dsupend=0.):                  
    cdef double end= Lo-10.
    cdef int test=1
    cdef double Th=Tho,Tv=Tvo,W=g*q1*Lo,Hmin_ok = 10000.,Hmin,xcoord,zcoord,F
    cdef Py_ssize_t ind
    cdef double middle = Lo*0.5-Csize, start = 10.
    cdef double s1 = middle 
    cdef double fo,go,rac1,rac2,rac3,rac4,dfTv,dfTh,dgTv,dgTh,coeff,h,k,Fx,Gz
    cdef unsigned int it = 0,step=100
    cdef unsigned int i, j
    cdef double err=1.0    
    while s1 > start:
        F = (0.5*((s1+Dsupdep)*q2+((Lo-s1)+Dsupend)*q3))*g + Fo
        #Newton Th Tv
        h = 100.0
        k = 100.0 
        it = 0   
        while fabs(h)>err and fabs(k)>err :
            fo = f_x(Th,Tv,Lo,EAo,W,F,s1,D) #fo
            go = f_z(Th,Tv,Lo,EAo,W,F,s1,H) #go
            rac1 = sqrt((Tv/Th)*(Tv/Th)+1)
            rac2 = sqrt(((Tv-F-W)/Th)*((Tv-F-W)/Th)+1)
            rac3 = sqrt(((Tv-F-W*s1/Lo)/Th)*((Tv-F-W*s1/Lo)/Th)+1)
            rac4 = sqrt(((Tv-W*s1/Lo)/Th)*((Tv-W*s1/Lo)/Th)+1)
            dfTv = Lo/W*(1.0/rac1-1.0/rac2+1.0/rac3-1.0/rac4)
            dfTh = df_dTh(Th,Tv,F,W,s1,Lo,EAo,rac1,rac2,rac3,rac4)
            dgTv = Lo/EAo + Lo/(W*Th)*(Tv/rac1-(Tv-F-W)/rac2+(Tv-F-W*s1/Lo)/rac3-(Tv-W*s1/Lo)/rac4)       
            dgTh = dg_dTh(Tv,Th,Lo,W,s1,F,rac1,rac2,rac3,rac4)
            coeff = 1.0/(dfTh*dgTv-dfTv*dgTh)        
            h = coeff*(-dgTv*fo+dfTv*go)
            k = coeff*(dgTh*fo-dfTh*go)
            Th = Th + h
            Tv = Tv + k
            it += 1
            if it>20: break             
        if fabs(f_x(Th,Tv,Lo,EAo,W,F,s1,D))+fabs(f_z(Th,Tv,Lo,EAo,W,F,s1,H))>0.03:
            test = 0
            break
        xcoord = Xup+fact*calcul_xs(Th,Tv,Lo,EAo,W,F,s1,s1)
        zcoord = Zup-calcul_zs(Th,Tv,Lo,EAo,W,F,s1,s1)
        ind = int(xcoord*2+0.5)
        Hmin = zcoord-(Alts[ind]+Hline_min)
        if Hmin<0 or Hmin+Hline_min>Hline_max or sqrt(Th*Th+Tv*Tv)>(Tmax+1000):
            test=0
            break
        else:
            Hmin_ok = double_min(Hmin_ok,Hmin)   
        s1 -= Csize
    if test:
        Th,Tv = Tho,Tvo
        middle = double_min(int(Lo*0.5+Csize),end)
        s1=middle
        while s1  < end :
            F = (0.5*((s1+Dsupdep)*q2+((Lo-s1)+Dsupend)*q3))*g + Fo
            #Newton Th Tv
            h = 100.0
            k = 100.0 
            it = 0
            while fabs(h)>err and fabs(k)>err :
                fo = f_x(Th,Tv,Lo,EAo,W,F,s1,D) #fo
                go = f_z(Th,Tv,Lo,EAo,W,F,s1,H) #go
                rac1 = sqrt((Tv/Th)*(Tv/Th)+1)
                rac2 = sqrt(((Tv-F-W)/Th)*((Tv-F-W)/Th)+1)
                rac3 = sqrt(((Tv-F-W*s1/Lo)/Th)*((Tv-F-W*s1/Lo)/Th)+1)
                rac4 = sqrt(((Tv-W*s1/Lo)/Th)*((Tv-W*s1/Lo)/Th)+1)
                dfTv = Lo/W*(1.0/rac1-1.0/rac2+1.0/rac3-1.0/rac4)
                dfTh = df_dTh(Th,Tv,F,W,s1,Lo,EAo,rac1,rac2,rac3,rac4)
                dgTv = Lo/EAo + Lo/(W*Th)*(Tv/rac1-(Tv-F-W)/rac2+(Tv-F-W*s1/Lo)/rac3-(Tv-W*s1/Lo)/rac4)       
                dgTh = dg_dTh(Tv,Th,Lo,W,s1,F,rac1,rac2,rac3,rac4)
                coeff = 1.0/(dfTh*dgTv-dfTv*dgTh)        
                h = coeff*(-dgTv*fo+dfTv*go)
                k = coeff*(dgTh*fo-dfTh*go)
                Th = Th + h
                Tv = Tv + k
                it += 1
                if it>20: break                 
            if fabs(f_x(Th,Tv,Lo,EAo,W,F,s1,D))+fabs(f_z(Th,Tv,Lo,EAo,W,F,s1,H))>0.03:
                test=0
                break   
            xcoord = Xup+fact*calcul_xs(Th,Tv,Lo,EAo,W,F,s1,s1)
            zcoord = Zup-calcul_zs(Th,Tv,Lo,EAo,W,F,s1,s1)
            ind = int(xcoord*2+0.5)
            Hmin = zcoord-(Alts[ind]+Hline_min)
            if Hmin<0 or Hmin+Hline_min>Hline_max or sqrt(Th*Th+Tv*Tv)>(Tmax+1000):
                test=0
                break
            else:
                Hmin_ok = double_min(Hmin_ok,Hmin)
            s1+= Csize
    if not test : Hmin_ok=-1.
    return Hmin_ok 



####Find Lomin modified in version 3.0
@cython.cdivision(True) 
@cython.boundscheck(False)
@cython.wraparound(False)   
cdef Find_Lomin(double D,double H,double Xup,double Zup,double fact,
		    np.ndarray[dtypef_t,ndim=1] Alts,double Fo,double Tmax,
	            double q1,double q2,double q3,double EAo,np.ndarray[dtypef_t,ndim=2] rastLosup,
                    np.ndarray[dtypef_t,ndim=2] rastTh,np.ndarray[dtypef_t,ndim=2] rastTv,
		    double Hline_min,double Hline_max,double Csize,double Dsupdep=0.,double Dsupend=0.):
    """
    Cherche Lomin telle que la tension du cable porteur avec la charge au milieu soit < Tmax
    ------
    Inputs:
    ------
    D       m       Distance horizontale du Span
    H       m       Distance verticale du Span
    Dsupdep m       Longueur supplementaire de cable sur le(s) span(s) situes du cote de la place de depot
    Dsupend m       Longueur supplementaire de cable sur le(s) span(s) situes du cote du support terminal
    Fo      N       Poids de la charge max + du charriot    
    q1      kg/m    Masse lineique du cable porteur
    q2      kg/m    Masse lineique du cable tracteur
    q3      kg/m    Masse lineique du cable retour

    """
    # Test if intermediate support are needed
    cdef double h = 100.0,err=1.0,error=50.
    cdef double k = 100.0
    cdef unsigned int it = 0,test=1,ind
    cdef double diag = sqrt(D*D+H*H)  
    cdef Py_ssize_t col = int_max(ceil(D-5)-1,0)
    cdef Py_ssize_t line = ceil(H)
    cdef double Lsup = rastLosup[line,col],Th=rastTh[line,col],Tv=rastTv[line,col] 
    cdef double fo,go,rac1,rac2,rac3,rac4,dfTv,dfTh,dgTv,dgTh,coeff,Fx,Gz
    cdef double Lo=Lsup+diag
    cdef double W= q1*g*Lo
    cdef double s1=0.5*Lo
    cdef double F=(0.5*((s1+Dsupdep)*q2+((Lo-s1)+Dsupend)*q3))*g + Fo
    cdef double Tcalc=sqrt(Th*Th+Tv*Tv)
    cdef double xcoord,zcoord,Hmin
    if npy_isnan(Th) or npy_isnan(Tv):
        test=0        
    if test:
        #Newton Th Tv
        while fabs(h)>err and fabs(k)>err :
            fo = f_x(Th,Tv,Lo,EAo,W,F,s1,D) #fo
            go = f_z(Th,Tv,Lo,EAo,W,F,s1,H) #go
            rac1 = sqrt((Tv/Th)*(Tv/Th)+1)
            rac2 = sqrt(((Tv-F-W)/Th)*((Tv-F-W)/Th)+1)
            rac3 = sqrt(((Tv-F-W*s1/Lo)/Th)*((Tv-F-W*s1/Lo)/Th)+1)
            rac4 = sqrt(((Tv-W*s1/Lo)/Th)*((Tv-W*s1/Lo)/Th)+1)
            dfTv = Lo/W*(1.0/rac1-1.0/rac2+1.0/rac3-1.0/rac4)
            dfTh = df_dTh(Th,Tv,F,W,s1,Lo,EAo,rac1,rac2,rac3,rac4)
            dgTv = Lo/EAo + Lo/(W*Th)*(Tv/rac1-(Tv-F-W)/rac2+(Tv-F-W*s1/Lo)/rac3-(Tv-W*s1/Lo)/rac4)  
            dgTh = dg_dTh(Tv,Th,Lo,W,s1,F,rac1,rac2,rac3,rac4)
            coeff = 1.0/(dfTh*dgTv-dfTv*dgTh)        
            h = coeff*(-dgTv*fo+dfTv*go)
            k = coeff*(dgTh*fo-dfTh*go)
            Th = Th + h
            Tv = Tv + k
            it += 1
            if double_min(Th,Tv) < 0:
                test=0
                break                       
            if it>20: 
                test=0
                break
        if test:
            # Find Lo so that Tcalc does not exceed Tmax
            Tcalc = sqrt(Th*Th+Tv*Tv)
            incr = 0.01
            signe = (Tcalc-Tmax)/fabs(Tcalc-Tmax)
            # Incrementation 
            while fabs(Tcalc-Tmax) > error and test:
                Lo += signe*incr
                W = q1*g*Lo
                s1 = Lo*0.5
                F =  (0.5*((s1+Dsupdep)*q2+((Lo-s1)+Dsupend)*q3))*g + Fo  
                h = 100.0
                k = 100.0
                it = 0
                #Newton Th Tv
                while fabs(h)>err and fabs(k)>err :
                    fo = f_x(Th,Tv,Lo,EAo,W,F,s1,D) #fo
                    go = f_z(Th,Tv,Lo,EAo,W,F,s1,H) #go
                    rac1 = sqrt((Tv/Th)*(Tv/Th)+1)
                    rac2 = sqrt(((Tv-F-W)/Th)*((Tv-F-W)/Th)+1)
                    rac3 = sqrt(((Tv-F-W*s1/Lo)/Th)*((Tv-F-W*s1/Lo)/Th)+1)
                    rac4 = sqrt(((Tv-W*s1/Lo)/Th)*((Tv-W*s1/Lo)/Th)+1)
                    dfTv = Lo/W*(1.0/rac1-1.0/rac2+1.0/rac3-1.0/rac4)
                    dfTh = df_dTh(Th,Tv,F,W,s1,Lo,EAo,rac1,rac2,rac3,rac4)
                    dgTv = Lo/EAo + Lo/(W*Th)*(Tv/rac1-(Tv-F-W)/rac2+(Tv-F-W*s1/Lo)/rac3-(Tv-W*s1/Lo)/rac4)       
                    dgTh = dg_dTh(Tv,Th,Lo,W,s1,F,rac1,rac2,rac3,rac4)
                    coeff = 1.0/(dfTh*dgTv-dfTv*dgTh)    
                    h = coeff*(-dgTv*fo+dfTv*go)
                    k = coeff*(dgTh*fo-dfTh*go)
                    Th = Th + h
                    Tv = Tv + k
                    it += 1
                    if double_min(Th,Tv) < 0:
                        test=0
                        break      
                    if it>20: 
                        test=0
                        break
                Tcalc = sqrt(Th*Th+Tv*Tv)
                if signe*(Tcalc-Tmax)<0: 
                    incr *= 0.1
                    signe *=-1.
                if fabs(Lo-sqrt(H*H+D*D))>100.:
                    test=0
                    break

    if test:
        F =  (0.5*((s1+Dsupdep)*q2+((Lo-s1)+Dsupend)*q3))*g + Fo  
        xcoord = Xup+fact*calcul_xs(Th,Tv,Lo,EAo,W,F,Lo*0.5,Lo*0.5)
        zcoord = Zup-calcul_zs(Th,Tv,Lo,EAo,W,F,Lo*0.5,Lo*0.5)
        ind = floor(xcoord*2+0.5)
        Hmin = zcoord-(Alts[ind]+Hline_min)
        if Hmin>=0:
            Hmin = check_Hlinemin(Alts,H,D,Lo,fact,Th,Tv,Xup,Zup,Fo,Tmax,Hline_min,Hline_max,q1,q2,q3,Csize,EAo,Dsupdep,Dsupend)
            if Hmin<0: 
                test=0    
        else:
            test=0
    return test,Lo,Th,Tv,Tcalc,F

####Add version 3.0 Process for test if span OK

@cython.cdivision(True) 
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef test_Span(np.ndarray[dtypef_t,ndim=2] Line,int pg,int posi,double Hg, double Hd,
                double Hline_min, double Hline_max,double slope_min,double slope_max,
                np.ndarray[dtypef_t,ndim=1] Alts,double Fo,double Tmax,double q1,
                double q2, double q3, double EAo,np.ndarray[dtypef_t,ndim=2] rastLosup,
                np.ndarray[dtypef_t,ndim=2] rastTh,np.ndarray[dtypef_t,ndim=2] rastTv,
                double Csize, double angle_intsup,double Dsupdep=0.,double slope_prev=-9999.):

    cdef unsigned int test=0
    cdef double D = Line[posi,0]-Line[pg,0]
    cdef double H = fabs(Line[pg,1]+Hg-(Line[posi,1]+Hd))  
    cdef double Xup,Zup,fact
    cdef double diag=0,slope=0,Lo=0,Th=0,Tv=0,F=0,Tcalc=0  
    if Line[pg,1]+Hg>=Line[posi,1]+Hd:
        Xup,Zup =Line[pg,0],Line[pg,1]+Hg
        fact = 1. 
    else:    
        Xup,Zup = Line[posi,0],Line[posi,1]+Hd
        fact = -1. 
    if check_droite(fact,H,D,Xup,Zup,Line,Hline_min,Hline_max,Tmax,q1,q2,q3,Fo,pg,posi,Dsupdep):  
        diag = sqrt(H*H+D*D)
        slope = -1*fact*atan(H/D)     
        if slope< slope_min or slope>slope_max:
            test=0
        else:
            # Check slopes around intermediate support
            if slope_prev>-9999 and (fabs(slope-slope_prev)>=angle_intsup or (slope*slope_prev<0 and fabs(slope-slope_prev)>=0.1)): 
                test=0
            else:
                test,Lo,Th,Tv,Tcalc,F = Find_Lomin(D,H,Xup,Zup,fact,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Hline_min,Hline_max,Csize,Dsupdep)
    return test,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F

@cython.boundscheck(False)
@cython.wraparound(False)  
cdef np.ndarray[dtypef_t,ndim=2] get_Tabis(np.ndarray[dtypef_t,ndim=2] Tab, int lineTab,int nbconfig,int intsup,int indmax):      
    cdef int i=0,j,col = (intsup-1)*14+13,colH=(intsup-1)*14+12
    cdef int idmax = 0,idmax2 = indmax+1,idline=0
    cdef int linemax = int_min(lineTab,nbconfig)
    cdef np.ndarray[dtypef_t,ndim=2] Tabis = np.zeros((linemax,Tab.shape[1]),dtype=np.float)
    cdef double Hmin = 100
    while i < linemax:
        for j in range(0,lineTab):
            if Tab[j,col]>=idmax and Tab[j,col]<idmax2:
                idmax=ceil(Tab[j,col])
                if Tab[j,colH]<Hmin:
                    Hmin=Tab[j,colH]
                    idline=j
        Tabis[i]=Tab[idline]
        i+=1
        idmax2=idmax
        idmax=0
        Hmin = 100
    return Tabis

@cython.boundscheck(False)
@cython.wraparound(False)      
cdef np.ndarray[dtypef_t,ndim=2] get_Tabis2(np.ndarray[dtypef_t,ndim=2] Tab, int lineTab,int nbconfig,int intsup,int indmax):      
    cdef int i=0,j,col = (intsup-1)*15+13,colH=(intsup-1)*15+12
    cdef int idmax = 0,idmax2 = indmax+1,idline=0
    cdef int linemax = int_min(lineTab,nbconfig)
    cdef np.ndarray[dtypef_t,ndim=2] Tabis = np.zeros((linemax,Tab.shape[1]),dtype=np.float)
    cdef double Hmin = 100
    while i < linemax:
        for j in range(0,lineTab):
            if Tab[j,col]>=idmax and Tab[j,col]<idmax2:
                idmax=ceil(Tab[j,col])
                if Tab[j,colH]<Hmin:
                    Hmin=Tab[j,colH]
                    idline=j
        Tabis[i]=Tab[idline]
        i+=1
        idmax2=idmax
        idmax=0
        Hmin = 100
    return Tabis
          
@cython.cdivision(True)  
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef np.ndarray[dtypef_t,ndim=2] OptPyl_Up(np.ndarray[dtypef_t,ndim=2] Line,np.ndarray[dtypef_t,ndim=1] Alts,np.ndarray[dtypef_t,ndim=2] Span,double Htower,
                                            double Hintsup,double Hend,double q1,double q2,double q3,double Fo,double Hline_min,double Hline_max,double Csize,
                                            double angle_intsup,double EAo,double E,double d,unsigned int sup_max,np.ndarray[dtypef_t,ndim=2] rastLosup,
                                            np.ndarray[dtypef_t,ndim=2] rastTh,np.ndarray[dtypef_t,ndim=2] rastTv,double Tmax,double LminSpan,double slope_min,
                                            double slope_max,double Lmax,unsigned int test_hfor,unsigned int nbconfig=10):               
    """
    Cable machine en haut
    Optimise le placement des pylones intermediaire et la hauteur de fixation sur chaque pylone sur un profil 
    
    """             
    cdef unsigned int indmax = Line.shape[0]-1, test=0,test0,test1,test2 
    cdef double D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F
    cdef int indminmulti, indmaxmulti,pg,posi,lineTab,lineTabis,i
    cdef double slope_prev,Dsupdep,newTmax,diff
    cdef np.ndarray[dtypef_t,ndim=2] Tab,Tabis  
    cdef int intsup,best,nblineTabis=1,indmin,p,c,col
    cdef double Hg,Hd,Hd2,Tdown,Hdmax,Hdmax2

    #################################
    # Begin without intermediate support
    #################################
    if test_hfor:
        Hd = Line[indmax,7]
    else:
        Hd = Hend    
    while Hd>1:    
        test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,indmax,Htower,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
        if test0:               
            Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
            Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
            Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
            Span[0,14],Span[0,15]=Hd,indmax
            test = 1
        else:
            break
        Hd-=1
    #################################
    # End without intermediate support 
    #################################
    #################################
    # Start Cut the line if no intermediate support allowed
    ################################# 
    if not test and sup_max==0:
        indminmulti = 0
        diff = 0.
        while diff <double_max(Csize,LminSpan):
            indminmulti += 1
            diff = Line[indminmulti,0]-Line[0,0]   
        for posi in range(indmax-1,indminmulti-1,-1):
            if test_hfor:
                Hd = Line[posi,7]
            else:
                Hd = Hend
            while Hd >1:
                test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,posi,Htower,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
                if test0:
                    Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
                    Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
                    Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
                    Span[0,14],Span[0,15]=Hd,posi
                    test=1
                else:
                    break
                Hd-=1
            if test:
                break
    #################################
    # End Cut the line if no intermediate support allowed
    ################################# 
    #################################
    # Start intermediate support position optimisation
    ################################# 
    if not test and sup_max>0: 
        #Get index respecting minimum length between two pylons
        indmaxmulti = indmax
        diff = 0.
        while diff <double_max(Csize,LminSpan) and indmaxmulti>0:
            indmaxmulti -= 1
            diff = Line[indmax,0]-Line[indmaxmulti,0]
        if indmaxmulti==0:
            test=1        
    if not test and sup_max>0:  
        indminmulti = 1       
        #Generate table to compare results 
        Tab = -9999*np.ones((indmaxmulti*nbconfig*100,14*(sup_max+1)),dtype=np.float)  
        lineTab=0
        Tabis = -9999*np.ones((1,14*(sup_max+1)),dtype=np.float)  
        lineTabis = 0
        # posi1 Tdown1 Dsupdep1 slope1 ....  
        ########################################
        #Optimize int sup implantation
        ########################################        
        test=1
        intsup=1
        newTmax = Tmax
        Dsupdep = 0
        pg=0
        Hg=Htower   
        best=0
        slope_prev=-9999
        indmin=indminmulti-1
        while intsup<=sup_max and not best:
            #Optimize
            for p in range(0,nblineTabis):
                if intsup>1:
                    lineTabis = p
                    col = (intsup-2)*14
                    indmin = ceil(Tabis[p,13+col])  
                    pg=indmin
                    newTmax=Tabis[p,10+col]
                    Dsupdep=0
                    Hg=Tabis[p,12+col]
                    for c in range(col+2,1,-14):
                        Dsupdep+=Tabis[p,c]    
                    slope_prev = Tabis[p,3+col]                    
                for posi in range(indmaxmulti,indmin,-1):
                    # Get info from nearest span 
                    Hdmax = Line[posi,7] 
                    Hd = ceil(Hline_min)
                    while Hd <= Hdmax:
                        test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                        if test1: 
                            Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                            Tab[lineTab,0:(intsup-1)*14]=Tabis[lineTabis,0:(intsup-1)*14]
                            Tab[lineTab,(intsup-1)*14:((intsup-1)*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,posi
                            lineTab+=1
                            # Get info from farest span
                            if test_hfor:
                                Hdmax2 = Line[indmax,7]
                            else:
                                Hdmax2 = Hend
                            Hd2=1
                            while Hd2 <=Hdmax2:
                                test2,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,posi,indmax,Hd,Hd2,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tdown,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,diag+Dsupdep,slope)
                                if test2:
                                    best=1
                                    Tab[lineTab,0:intsup*14]=Tab[lineTab-1,0:intsup*14]                         
                                    Tab[lineTab,intsup*14:(intsup*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd2,indmax
                                    break
                                Hd2+=1
                            if best:
                                break
                        Hd+=1
                    if best:
                        break                            
                if best:                    
                    break
            if not best:                
                if lineTab==0:
                    test=0
                    intsup-=2
                    if Tabis[0,0]>0:                    
                        Tab[lineTab] = Tabis[0]
                    else:
                        Tab[lineTab] *= 0
                    break
                Tabis = get_Tabis(Tab,lineTab,nbconfig,intsup,indmax)
                nblineTabis=Tabis.shape[0]
                if nblineTabis>0:
                    Tab = -9999*np.ones((indmaxmulti*Tabis.shape[0]*nbconfig*100,14*(sup_max+1)),dtype=np.float)  
                    intsup+=1  
                    lineTab=0
                else:
                    test=0
                    intsup=-2
                    break
        ######## cut the line at the farest position if sup_max>0
        if not best and test:            
            lineTab=0
            for p in range(0,nblineTabis):
                lineTabis = p
                col = (intsup-2)*14
                pg = ceil(Tabis[p,13+col]) 
                Hg=Tabis[p,12+col]
                indminmulti = pg
                diff = 0.
                while diff <double_max(Csize,LminSpan):
                    indminmulti += 1
                    diff = Line[indminmulti,0]-Line[pg,0]  
                newTmax=Tabis[p,10+col]
                Dsupdep=0
                for c in range(col+2,1,-14):
                    Dsupdep+=Tabis[p,c]    
                slope_prev = Tabis[p,3+col]
                for posi in range(indmax-1,indminmulti-1,-1):
                    # Get info from nearest span
                    if test_hfor:
                        Hdmax = Line[indmax,7]
                    else:
                        Hdmax = Hend
                    Hd=1
                    while Hd <=Hdmax:
                        test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                        if test1: 
                            Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                            Tab[lineTab,0:(intsup-1)*14]=Tabis[lineTabis,0:(intsup-1)*14]
                            Tab[lineTab,(intsup-1)*14:((intsup-1)*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,posi
                            lineTab+=1
                            break
                        Hd+=1                    
            if lineTab==0:
                intsup-=2
                lineTab=0                                   
                Tab[lineTab] = Tabis[0]
            else:
                Tab = get_Tabis(Tab,lineTab,1,intsup,indmax)                
                lineTab=0       
                intsup=sup_max 
        ######## Save Span carac 
        for i in range(0,intsup+1):
            col = i*14            
            Span[i,[0,1,2,3,4,5,6,7,8,9,10,11,14,15]]=Tab[lineTab,col:col+14]
    return Span                 

@cython.cdivision(True)  
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef np.ndarray[dtypef_t,ndim=2] OptPyl_Up_NoH(np.ndarray[dtypef_t,ndim=2] Line,np.ndarray[dtypef_t,ndim=1] Alts,np.ndarray[dtypef_t,ndim=2] Span,double Htower,
                                            double Hintsup,double Hend,double q1,double q2,double q3,double Fo,double Hline_min,double Hline_max,double Csize,
                                            double angle_intsup,double EAo,double E,double d,unsigned int sup_max,np.ndarray[dtypef_t,ndim=2] rastLosup,
                                            np.ndarray[dtypef_t,ndim=2] rastTh,np.ndarray[dtypef_t,ndim=2] rastTv,double Tmax,double LminSpan,double slope_min,
                                            double slope_max,double Lmax,unsigned int test_hfor,unsigned int nbconfig=10):               
    """
    Cable machine en haut
    Optimise le placement des pylones intermediaire sans bouger la hauteur de fixation du cable porteur pour chaque pylone sur un profil 
    
    """                
    cdef unsigned int indmax = Line.shape[0]-1, test=0,test0,test1,test2 
    cdef double D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F
    cdef int indminmulti, indmaxmulti,pg,posi,lineTab,lineTabis,i
    cdef double slope_prev,Dsupdep,newTmax,diff
    cdef np.ndarray[dtypef_t,ndim=2] Tab,Tabis  
    cdef int intsup,best,nblineTabis=1,indmin,p,c,col
    cdef double Hg,Hd,Hd2,Tdown

    #################################
    # Begin without intermediate support
    #################################
    if test_hfor:
        Hd = Line[indmax,7]
    else:
        Hd = Hend   
    test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,indmax,Htower,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
    if test0:               
        Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
        Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
        Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
        Span[0,14],Span[0,15]=Hd,indmax
        test = 1
    #################################
    # End without intermediate support 
    #################################
    #################################
    # Start Cut the line if no intermediate support allowed
    ################################# 
    if not test and sup_max==0:
        indminmulti = 0
        diff = 0.
        while diff <double_max(Csize,LminSpan):
            indminmulti += 1
            diff = Line[indminmulti,0]-Line[0,0]   
        for posi in range(indmax-1,indminmulti-1,-1):
            if test_hfor:
                Hd = Line[posi,7]
            else:
                Hd = Hend            
            test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,posi,Htower,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
            if test0:
                Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
                Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
                Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
                Span[0,14],Span[0,15]=Hd,posi
                test=1                
                break
    #################################
    # End Cut the line if no intermediate support allowed
    ################################# 
    #################################
    # Start intermediate support position optimisation
    ################################# 
    if not test and sup_max>0: 
        #Get index respecting minimum length between two pylons
        indmaxmulti = indmax
        diff = 0.
        while diff <double_max(Csize,LminSpan) and indmaxmulti>0:
            indmaxmulti -= 1
            diff = Line[indmax,0]-Line[indmaxmulti,0]
        if indmaxmulti==0:
            test=1        
    if not test and sup_max>0:  
        indminmulti = 1       
        #Generate table to compare results 
        Tab = -9999*np.ones((indmaxmulti*100*nbconfig,14*(sup_max+1)),dtype=np.float)  
        lineTab=0
        Tabis = -9999*np.ones((1,14*(sup_max+1)),dtype=np.float)  
        lineTabis = 0
        # posi1 Tdown1 Dsupdep1 slope1 ....  
        ########################################
        #Optimize int sup implantation
        ########################################        
        test=1
        intsup=1
        newTmax = Tmax
        Dsupdep = 0
        pg=0
        Hg=Htower   
        best=0
        slope_prev=-9999
        indmin=indminmulti-1
        while intsup<=sup_max and not best:
            #Optimize
            for p in range(0,nblineTabis):
                if intsup>1:
                    lineTabis = p
                    col = (intsup-2)*14
                    indmin = ceil(Tabis[p,13+col])  
                    pg=indmin
                    newTmax=Tabis[p,10+col]
                    Dsupdep=0
                    Hg=Tabis[p,12+col]
                    for c in range(col+2,1,-14):
                        Dsupdep+=Tabis[p,c]    
                    slope_prev = Tabis[p,3+col]                    
                for posi in range(indmaxmulti,indmin,-1):
                    # Get info from nearest span 
                    Hd = Line[posi,7]                 
                    test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                    if test1: 
                        Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                        Tab[lineTab,0:(intsup-1)*14]=Tabis[lineTabis,0:(intsup-1)*14]
                        Tab[lineTab,(intsup-1)*14:((intsup-1)*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,posi
                        lineTab+=1
                        # Get info from farest span
                        if test_hfor:
                            Hd2 = Line[indmax,7]
                        else:
                            Hd2 = Hend
                        test2,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,posi,indmax,Hd,Hd2,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tdown,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,diag+Dsupdep,slope)
                        if test2:
                            best=1
                            Tab[lineTab,0:intsup*14]=Tab[lineTab-1,0:intsup*14]                         
                            Tab[lineTab,intsup*14:(intsup*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd2,indmax
                            break            
                if best:                    
                    break
            if not best:                
                if lineTab==0:
                    test=0
                    intsup-=2
                    if Tabis[0,0]>0:                    
                        Tab[lineTab] = Tabis[0]
                    else:
                        Tab[lineTab] *= 0
                    break
                Tabis = get_Tabis(Tab,lineTab,nbconfig,intsup,indmax)
                nblineTabis=Tabis.shape[0]
                if nblineTabis>0:
                    Tab = -9999*np.ones((indmaxmulti*Tabis.shape[0]*100*nbconfig,14*(sup_max+1)),dtype=np.float)  
                    intsup+=1  
                    lineTab=0
                else:
                    test=0
                    intsup=-2
                    break
        ######## cut the line at the farest position if sup_max>0
        if not best and test:            
            lineTab=0
            for p in range(0,nblineTabis):
                lineTabis = p
                col = (intsup-2)*14
                pg = ceil(Tabis[p,13+col]) 
                Hg=Tabis[p,12+col]
                indminmulti = pg
                diff = 0.
                while diff <double_max(Csize,LminSpan):
                    indminmulti += 1
                    diff = Line[indminmulti,0]-Line[pg,0]  
                newTmax=Tabis[p,10+col]
                Dsupdep=0
                for c in range(col+2,1,-14):
                    Dsupdep+=Tabis[p,c]    
                slope_prev = Tabis[p,3+col]
                for posi in range(indmax-1,indminmulti-1,-1):
                    # Get info from nearest span
                    if test_hfor:
                        Hd = Line[indmax,7]
                    else:
                        Hd = Hend             
                    test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                    if test1: 
                        Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                        Tab[lineTab,0:(intsup-1)*14]=Tabis[lineTabis,0:(intsup-1)*14]
                        Tab[lineTab,(intsup-1)*14:((intsup-1)*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,posi
                        lineTab+=1
                        break               
            if lineTab==0:
                intsup-=2
                lineTab=0                                   
                Tab[lineTab] = Tabis[0]
            else:
                Tab = get_Tabis(Tab,lineTab,1,intsup,indmax)                
                lineTab=0       
                intsup=sup_max 
        ######## Save Span carac 
        for i in range(0,intsup+1):
            col = i*14            
            Span[i,[0,1,2,3,4,5,6,7,8,9,10,11,14,15]]=Tab[lineTab,col:col+14]
    return Span        

@cython.cdivision(True)  
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef np.ndarray[dtypef_t,ndim=2] OptPyl_Down_init(np.ndarray[dtypef_t,ndim=2] Line,np.ndarray[dtypef_t,ndim=1] Alts,np.ndarray[dtypef_t,ndim=2] Span,double Htower,
                                            double Hintsup,double Hend,double q1,double q2,double q3,double Fo,double Hline_min,double Hline_max,double Csize,
                                            double angle_intsup,double EAo,double E,double d,unsigned int sup_max,np.ndarray[dtypef_t,ndim=2] rastLosup,
                                            np.ndarray[dtypef_t,ndim=2] rastTh,np.ndarray[dtypef_t,ndim=2] rastTv,double Tmax,double LminSpan,double slope_min,
                                            double slope_max,double Lmax,unsigned int test_hfor,unsigned int nbconfig=5):               
    """
    Cable machine en bas
    Permet de recuperer la partie de profil ou il est possible de tendre un cable (avec hauteur de cable porteur variable)
    
    """               
    cdef unsigned int indmax = Line.shape[0]-1, test=0,test0,test1,test2 
    cdef double D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F
    cdef int indminmulti, indmaxmulti,pg,posi,lineTab,lineTabis,i
    cdef double slope_prev,Dsupdep,diff
    cdef np.ndarray[dtypef_t,ndim=2] Tab,Tabis  
    cdef int intsup,best,nblineTabis=1,indmin,p,c,col
    cdef double Hg,Hd,Hd2,Tdown,Hdmax,Hdmax2

    #################################
    # Begin without intermediate support
    #################################
    if test_hfor:
        Hd = Line[indmax,7]
    else:
        Hd = Hend    
    while Hd>1:    
        test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,indmax,Htower,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
        if test0:               
            Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
            Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
            Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
            Span[0,14],Span[0,15]=Hd,indmax
            test = 1  
        else:
            break
        Hd-=1
    #################################
    # End without intermediate support 
    #################################
    #################################
    # Start Cut the line if no intermediate support allowed
    ################################# 
    if not test and sup_max==0:
        indminmulti = 0
        diff = 0.
        while diff <double_max(Csize,LminSpan):
            indminmulti += 1
            diff = Line[indminmulti,0]-Line[0,0]   
        for posi in range(indmax-1,indminmulti-1,-1):
            if test_hfor:
                Hd = Line[posi,7]
            else:
                Hd = Hend
            while Hd >1:
                test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,posi,Htower,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
                if test0:
                    Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
                    Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
                    Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
                    Span[0,14],Span[0,15]=Hd,posi
                    test=1
                else:
                    break
                Hd-=1
            if test:
                break
    #################################
    # End Cut the line if no intermediate support allowed
    ################################# 
    #################################
    # Start intermediate support position optimisation
    ################################# 
    if not test and sup_max>0: 
        #Get index respecting minimum length between two pylons
        indmaxmulti = indmax
        diff = 0.
        while diff <double_max(Csize,LminSpan) and indmaxmulti>0:
            indmaxmulti -= 1
            diff = Line[indmax,0]-Line[indmaxmulti,0]
        if indmaxmulti==0:
            test=1        
    if not test and sup_max>0:  
        #Generate table to compare results 
        Tab = -9999*np.ones((indmaxmulti*100*nbconfig,14*(sup_max+1)),dtype=np.float)  
        lineTab=0
        Tabis = -9999*np.ones((1,14*(sup_max+1)),dtype=np.float)  
        lineTabis = 0
        # posi1 Tdown1 Dsupdep1 slope1 ....  
        ########################################
        #Optimize int sup implantation
        ########################################        
        test=1
        intsup=1
        newTmax = Tmax
        Dsupdep = 0
        pg=0
        Hg=Htower   
        best=0
        slope_prev=-9999
        indmin=0
        while intsup<=sup_max and not best:
            #Optimize
            for p in range(0,nblineTabis):
                if intsup>1:
                    lineTabis = p
                    col = (intsup-2)*14
                    indmin = ceil(Tabis[p,13+col])  
                    pg=indmin
                    Dsupdep=0
                    Hg=Tabis[p,12+col]
                    for c in range(col+2,1,-14):
                        Dsupdep+=Tabis[p,c]    
                    slope_prev = Tabis[p,3+col]
                for posi in range(indmaxmulti,indmin,-1):
                    # Get info from nearest span                    
                    Hdmax = Line[posi,7] 
                    Hd = ceil(Hline_min)
                    while Hd <= Hdmax:
                        test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                        if test1: 
                            Tab[lineTab,0:(intsup-1)*14]=Tabis[lineTabis,0:(intsup-1)*14]
                            Tab[lineTab,(intsup-1)*14:((intsup-1)*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,posi
                            lineTab+=1
                            # Get info from farest span
                            if test_hfor:
                                Hdmax2 = Line[indmax,7]
                            else:
                                Hdmax2 = Hend
                            Hd2=1
                            while Hd2 <=Hdmax2:
                                test2,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,posi,indmax,Hd,Hd2,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,diag+Dsupdep,slope)
                                if test2:
                                    best=1
                                    Tab[lineTab,0:intsup*14]=Tab[lineTab-1,0:intsup*14]                         
                                    Tab[lineTab,intsup*14:(intsup*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd2,indmax
                                    break
                                Hd2+=1
                            if best:
                                break
                        Hd+=1
                    if best:
                        break                            
                if best:                    
                    break
            if not best:                
                if lineTab==0:
                    test=0
                    intsup-=2
                    if Tabis[0,0]>0:                    
                        Tab[lineTab] = Tabis[0]
                    else:
                        Tab[lineTab] *= 0
                    break
                Tabis = get_Tabis(Tab,lineTab,nbconfig,intsup,indmax)
                nblineTabis=Tabis.shape[0]
                if nblineTabis>0:
                    Tab = -9999*np.ones((indmaxmulti*Tabis.shape[0]*100*nbconfig,14*(sup_max+1)),dtype=np.float)  
                    intsup+=1  
                    lineTab=0
                else:
                    test=0
                    intsup=-2
                    break
        ######## cut the line at the farest position if sup_max>0
        if not best and test:            
            lineTab=0
            for p in range(0,nblineTabis):
                lineTabis = p
                col = (intsup-2)*14
                pg = ceil(Tabis[p,13+col]) 
                Hg=Tabis[p,12+col]
                indminmulti = pg
                diff = 0.
                while diff <double_max(Csize,LminSpan):
                    indminmulti += 1
                    diff = Line[indminmulti,0]-Line[pg,0]                  
                Dsupdep=0
                for c in range(col+2,1,-14):
                    Dsupdep+=Tabis[p,c]    
                slope_prev = Tabis[p,3+col]
                for posi in range(indmax-1,indminmulti-1,-1):
                    # Get info from nearest span
                    if test_hfor:
                        Hdmax = Line[indmax,7]
                    else:
                        Hdmax = Hend
                    Hd=1
                    while Hd <=Hdmax:
                        test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                        if test1: 
                            Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                            Tab[lineTab,0:(intsup-1)*14]=Tabis[lineTabis,0:(intsup-1)*14]
                            Tab[lineTab,(intsup-1)*14:((intsup-1)*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,posi
                            lineTab+=1
                            break
                        Hd+=1                    
            if lineTab==0:
                intsup-=2
                lineTab=0                                   
                Tab[lineTab] = Tabis[0]
            else:
                Tab = get_Tabis(Tab,lineTab,1,intsup,indmax)                
                lineTab=0       
                intsup=sup_max 
        ######## Save Span carac 
        for i in range(0,intsup+1):
            col = i*14            
            Span[i,[0,1,2,3,4,5,6,7,8,9,10,11,14,15]]=Tab[lineTab,col:col+14]
    return Span    

@cython.cdivision(True)  
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef np.ndarray[dtypef_t,ndim=2] OptPyl_Down_init_NoH(np.ndarray[dtypef_t,ndim=2] Line,np.ndarray[dtypef_t,ndim=1] Alts,np.ndarray[dtypef_t,ndim=2] Span,double Htower,
                                            double Hintsup,double Hend,double q1,double q2,double q3,double Fo,double Hline_min,double Hline_max,double Csize,
                                            double angle_intsup,double EAo,double E,double d,unsigned int sup_max,np.ndarray[dtypef_t,ndim=2] rastLosup,
                                            np.ndarray[dtypef_t,ndim=2] rastTh,np.ndarray[dtypef_t,ndim=2] rastTv,double Tmax,double LminSpan,double slope_min,
                                            double slope_max,double Lmax,unsigned int test_hfor,unsigned int nbconfig=5):               
    """
    Cable machine en bas
    Permet de recuperer la partie de profil ou il est possible de tendre un cable (avec hauteur de cable porteur fixe)
    
    """                
    cdef unsigned int indmax = Line.shape[0]-1, test=0,test0,test1,test2 
    cdef double D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F
    cdef int indminmulti, indmaxmulti,pg,posi,lineTab,lineTabis,i
    cdef double slope_prev,Dsupdep,diff
    cdef np.ndarray[dtypef_t,ndim=2] Tab,Tabis  
    cdef int intsup,best,nblineTabis=1,indmin,p,c,col
    cdef double Hg,Hd,Hd2,Tdown

    #################################
    # Begin without intermediate support
    #################################
    if test_hfor:
        Hd = Line[indmax,7]
    else:
        Hd = Hend          
    test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,indmax,Htower,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
    if test0:               
        Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
        Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
        Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
        Span[0,14],Span[0,15]=Hd,indmax
        test = 1  

    #################################
    # End without intermediate support 
    #################################
    #################################
    # Start Cut the line if no intermediate support allowed
    ################################# 
    if not test and sup_max==0:
        indminmulti = 0
        diff = 0.
        while diff <double_max(Csize,LminSpan):
            indminmulti += 1
            diff = Line[indminmulti,0]-Line[0,0]   
        for posi in range(indmax-1,indminmulti-1,-1):
            if test_hfor:
                Hd = Line[posi,7]
            else:
                Hd = Hend
            test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,posi,Htower,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
            if test0:
                Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
                Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
                Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
                Span[0,14],Span[0,15]=Hd,posi
                test=1
                break
    #################################
    # End Cut the line if no intermediate support allowed
    ################################# 
    #################################
    # Start intermediate support position optimisation
    ################################# 
    if not test and sup_max>0: 
        #Get index respecting minimum length between two pylons
        indmaxmulti = indmax
        diff = 0.
        while diff <double_max(Csize,LminSpan) and indmaxmulti>0:
            indmaxmulti -= 1
            diff = Line[indmax,0]-Line[indmaxmulti,0]
        if indmaxmulti==0:
            test=1        
    if not test and sup_max>0:  
        #Generate table to compare results 
        Tab = -9999*np.ones((indmaxmulti*100*nbconfig,14*(sup_max+1)),dtype=np.float)  
        lineTab=0
        Tabis = -9999*np.ones((1,14*(sup_max+1)),dtype=np.float)  
        lineTabis = 0
        # posi1 Tdown1 Dsupdep1 slope1 ....  
        ########################################
        #Optimize int sup implantation
        ########################################        
        test=1
        intsup=1
        newTmax = Tmax
        Dsupdep = 0
        pg=0
        Hg=Htower   
        best=0
        slope_prev=-9999
        indmin=0
        while intsup<=sup_max and not best:
            #Optimize
            for p in range(0,nblineTabis):
                if intsup>1:
                    lineTabis = p
                    col = (intsup-2)*14
                    indmin = ceil(Tabis[p,13+col])  
                    pg=indmin
                    Dsupdep=0
                    Hg=Tabis[p,12+col]
                    for c in range(col+2,1,-14):
                        Dsupdep+=Tabis[p,c]    
                    slope_prev = Tabis[p,3+col]                    
                for posi in range(indmaxmulti,indmin,-1):
                    # Get info from nearest span                    
                    Hd = Line[posi,7]                       
                    test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                    if test1: 
                        Tab[lineTab,0:(intsup-1)*14]=Tabis[lineTabis,0:(intsup-1)*14]
                        Tab[lineTab,(intsup-1)*14:((intsup-1)*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,posi
                        lineTab+=1
                        # Get info from farest span
                        if test_hfor:
                            Hd2 = Line[indmax,7]
                        else:
                            Hd2 = Hend
                        test2,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,posi,indmax,Hd,Hd2,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,diag+Dsupdep,slope)
                        if test2:
                            best=1
                            Tab[lineTab,0:intsup*14]=Tab[lineTab-1,0:intsup*14]                         
                            Tab[lineTab,intsup*14:(intsup*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd2,indmax
                            break                                         
                if best:                    
                    break
            if not best:                
                if lineTab==0:
                    test=0
                    intsup-=2
                    if Tabis[0,0]>0:                    
                        Tab[lineTab] = Tabis[0]
                    else:
                        Tab[lineTab] *= 0
                    break
                Tabis = get_Tabis(Tab,lineTab,nbconfig,intsup,indmax)
                nblineTabis=Tabis.shape[0]
                if nblineTabis>0:
                    Tab = -9999*np.ones((indmaxmulti*Tabis.shape[0]*100*nbconfig,14*(sup_max+1)),dtype=np.float)  
                    intsup+=1  
                    lineTab=0
                else:
                    test=0
                    intsup=-2
                    break
        ######## cut the line at the farest position if sup_max>0
        if not best and test:            
            lineTab=0
            for p in range(0,nblineTabis):
                lineTabis = p
                col = (intsup-2)*14
                pg = ceil(Tabis[p,13+col]) 
                Hg=Tabis[p,12+col]
                indminmulti = pg
                diff = 0.
                while diff <double_max(Csize,LminSpan):
                    indminmulti += 1
                    diff = Line[indminmulti,0]-Line[pg,0]                  
                Dsupdep=0
                for c in range(col+2,1,-14):
                    Dsupdep+=Tabis[p,c]    
                slope_prev = Tabis[p,3+col]
                for posi in range(indmax-1,indminmulti-1,-1):
                    # Get info from nearest span
                    if test_hfor:
                        Hd = Line[indmax,7]
                    else:
                        Hd = Hend                    
                    test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                    if test1: 
                        Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                        Tab[lineTab,0:(intsup-1)*14]=Tabis[lineTabis,0:(intsup-1)*14]
                        Tab[lineTab,(intsup-1)*14:((intsup-1)*14+14)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,posi
                        lineTab+=1            
            if lineTab==0:
                intsup-=2
                lineTab=0                                   
                Tab[lineTab] = Tabis[0]
            else:
                Tab = get_Tabis(Tab,lineTab,1,intsup,indmax)                
                lineTab=0       
                intsup=sup_max 
        ######## Save Span carac 
        for i in range(0,intsup+1):
            col = i*14            
            Span[i,[0,1,2,3,4,5,6,7,8,9,10,11,14,15]]=Tab[lineTab,col:col+14]
    return Span    

@cython.cdivision(True)  
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef np.ndarray[dtypef_t,ndim=2] OptPyl_Up2(np.ndarray[dtypef_t,ndim=2] Line,np.ndarray[dtypef_t,ndim=1] Alts,np.ndarray[dtypef_t,ndim=2] Span,double Htower,
                                            double Hintsup,double Hend,double q1,double q2,double q3,double Fo,double Hline_min,double Hline_max,double Csize,
                                            double angle_intsup,double EAo,double E,double d,unsigned int sup_max,np.ndarray[dtypef_t,ndim=2] rastLosup,
                                            np.ndarray[dtypef_t,ndim=2] rastTh,np.ndarray[dtypef_t,ndim=2] rastTv,double Tmax,double LminSpan,double slope_min,
                                            double slope_max,double Lmax,unsigned int test_hfor,unsigned int nbconfig=10):               
    """
    Cable machine en bas
    Optimise le placement des pylones intermediaire et la hauteur de fixation du cable porteur pour chaque pylone sur un profil 
        
    """                
    cdef unsigned int indmax = Line.shape[0]-1, test=0,test0,test1,test2 
    cdef double D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F
    cdef int indminmulti, indmaxmulti,pg,posi,lineTab,lineTabis,i
    cdef double slope_prev,Dsupdep,newTmax,diff
    cdef np.ndarray[dtypef_t,ndim=2] Tab,Tabis  
    cdef int intsup,best,nblineTabis=1,indmin,p,c,col
    cdef double Hg,Hd,Tdown,Hgmax,Hdmax,Hginit

    #################################
    # Begin without intermediate support
    #################################
    if test_hfor:
        Hg = Line[indmax,7]
    else:
        Hg = Hend    
    while Hg>1:    
        test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,indmax,Hg,Htower,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
        if test0:               
            Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
            Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
            Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
            Span[0,14],Span[0,15]=Hg,indmax
            test = 1
        else:
            break
        Hg-=1
    #################################
    # End without intermediate support 
    #################################
    #################################
    # Start Cut the line if no intermediate support allowed
    ################################# 
    if not test and sup_max==0:
        indminmulti = 0
        diff = 0.
        while diff <double_max(Csize,LminSpan):
            indminmulti += 1
            diff = Line[indminmulti,0]-Line[0,0]   
        for posi in range(indmax-1,indminmulti-1,-1):
            if test_hfor:
                Hg = Line[posi,7]
            else:
                Hg = Hend
            while Hg >1:
                test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,posi,Hg,Htower,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
                if test0:
                    Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
                    Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
                    Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
                    Span[0,14],Span[0,15]=Hg,posi
                    test=1
                else:
                    break
                Hg-=1
            if test:
                break
    #################################
    # End Cut the line if no intermediate support allowed
    ################################# 
    #################################
    # Start intermediate support position optimisation
    ################################# 
    if not test and sup_max>0: 
        #Get index respecting minimum length between two pylons
        indmaxmulti = indmax
        diff = 0.
        while diff <double_max(Csize,LminSpan) and indmaxmulti>0:
            indmaxmulti -= 1
            diff = Line[indmax,0]-Line[indmaxmulti,0]
        if indmaxmulti==0:
            test=1        
    if not test and sup_max>0:  
        #Generate table to compare results 
        Tab = -9999*np.ones((indmaxmulti*100*nbconfig,15*(sup_max+1)),dtype=np.float)  
        lineTab=0
        Tabis = -9999*np.ones((1,15*(sup_max+1)),dtype=np.float)  
        lineTabis = 0
        # posi1 Tdown1 Dsupdep1 slope1 ....  
        ########################################
        #Optimize int sup implantation
        ########################################        
        test=1
        intsup=1
        newTmax = Tmax
        Dsupdep = 0
        pg=0
        if test_hfor:
            Hgmax = Line[0,7]
        else:
            Hgmax = Hend
        Hginit=1
        best=0
        slope_prev=-9999
        indmin=0
        while intsup<=sup_max and not best:
            #Optimize
            for p in range(0,nblineTabis):
                if intsup>1:
                    lineTabis = p
                    col = (intsup-2)*15
                    indmin = ceil(Tabis[p,13+col])  
                    pg=indmin
                    newTmax=Tabis[p,10+col]
                    Dsupdep=0
                    Hgmax=Tabis[p,14+col]
                    Hginit=Hgmax                    
                    for c in range(col+2,1,-15):
                        Dsupdep+=Tabis[p,c]    
                    slope_prev = Tabis[p,3+col]                    
                for posi in range(indmaxmulti,indmin,-1):
                    # Get info from nearest span 
                    Hdmax = Line[posi,7]
                    Hg=Hginit
                    while Hg<=Hgmax:  
                        Hd = ceil(Hline_min)
                        while Hd <= Hdmax:
                            test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                            if test1: 
                                Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                                Tab[lineTab,0:(intsup-1)*15]=Tabis[lineTabis,0:(intsup-1)*15]
                                Tab[lineTab,(intsup-1)*15:((intsup-1)*15+15)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hg,posi,Hd
                                lineTab+=1
                                # Get info from farest span
                                test2,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,posi,indmax,Hd,Htower,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tdown,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,diag+Dsupdep,slope)
                                if test2:
                                    best=1
                                    Tab[lineTab,0:intsup*15]=Tab[lineTab-1,0:intsup*15]                         
                                    Tab[lineTab,intsup*15:(intsup*15+15)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,indmax,Htower
                                    break
                            Hd+=1
                        if best:
                            break
                        Hg+=1                    
                    if best:
                        break                           
                if best:                    
                    break
            if not best:                
                if lineTab==0:
                    test=0
                    intsup-=2
                    if Tabis[0,0]>0:                    
                        Tab[lineTab] = Tabis[0]
                    else:
                        Tab[lineTab] *= 0
                    break
                Tabis = get_Tabis2(Tab,lineTab,nbconfig,intsup,indmax)
                nblineTabis=Tabis.shape[0]
                if nblineTabis>0:
                    Tab = -9999*np.ones((indmaxmulti*Tabis.shape[0]*100*nbconfig,15*(sup_max+1)),dtype=np.float)  
                    intsup+=1  
                    lineTab=0
                else:
                    test=0
                    intsup=-2
                    break
        ######## cut the line at the farest position if sup_max>0
        if not best and test:            
            lineTab=0
            for p in range(0,nblineTabis):
                lineTabis = p
                col = (intsup-2)*15
                pg = ceil(Tabis[p,13+col]) 
                Hg=Tabis[p,14+col]
                indminmulti = pg
                diff = 0.
                while diff <double_max(Csize,LminSpan):
                    indminmulti += 1
                    diff = Line[indminmulti,0]-Line[pg,0]  
                newTmax=Tabis[p,10+col]
                Dsupdep=0
                for c in range(col+2,1,-15):
                    Dsupdep+=Tabis[p,c]    
                slope_prev = Tabis[p,3+col]
                for posi in range(indmax-1,indminmulti-1,-1):
                    # Get info from nearest span                    
                    test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Htower,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                    if test1: 
                        Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                        Tab[lineTab,0:(intsup-1)*15]=Tabis[lineTabis,0:(intsup-1)*15]
                        Tab[lineTab,(intsup-1)*15:((intsup-1)*15+15)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hg,posi,Htower
                        lineTab+=1             
            if lineTab==0:
                intsup-=2
                lineTab=0                                   
                Tab[lineTab] = Tabis[0]
            else:
                Tab = get_Tabis2(Tab,lineTab,1,intsup,indmax)                
                lineTab=0       
                intsup=sup_max 
        ######## Save Span carac 
        for i in range(0,intsup+1):
            col = i*15            
            Span[i,[0,1,2,3,4,5,6,7,8,9,10,11,14,15]]=Tab[lineTab,col:col+14]
    return Span      

@cython.cdivision(True)  
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef np.ndarray[dtypef_t,ndim=2] OptPyl_Up2_NoH(np.ndarray[dtypef_t,ndim=2] Line,np.ndarray[dtypef_t,ndim=1] Alts,np.ndarray[dtypef_t,ndim=2] Span,double Htower,
                                            double Hintsup,double Hend,double q1,double q2,double q3,double Fo,double Hline_min,double Hline_max,double Csize,
                                            double angle_intsup,double EAo,double E,double d,unsigned int sup_max,np.ndarray[dtypef_t,ndim=2] rastLosup,
                                            np.ndarray[dtypef_t,ndim=2] rastTh,np.ndarray[dtypef_t,ndim=2] rastTv,double Tmax,double LminSpan,double slope_min,
                                            double slope_max,double Lmax,unsigned int test_hfor,unsigned int nbconfig=10):               
    """
    Cable machine en bas
    Optimise le placement des pylones intermediaire sans bouger la hauteur de fixation du cable porteur pour chaque pylone sur un profil 
        
    """               
    cdef unsigned int indmax = Line.shape[0]-1, test=0,test0,test1,test2 
    cdef double D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F
    cdef int indminmulti, indmaxmulti,pg,posi,lineTab,lineTabis,i
    cdef double slope_prev,Dsupdep,newTmax,diff
    cdef np.ndarray[dtypef_t,ndim=2] Tab,Tabis  
    cdef int intsup,best,nblineTabis=1,indmin,p,c,col
    cdef double Hg,Hd,Tdown,Hginit

    #################################
    # Begin without intermediate support
    #################################
    if test_hfor:
        Hg = Line[indmax,7]
    else:
        Hg = Hend         
    test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,indmax,Hg,Htower,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
    if test0:               
        Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
        Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
        Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
        Span[0,14],Span[0,15]=Hg,indmax
        test = 1        
    #################################
    # End without intermediate support 
    #################################
    #################################
    # Start Cut the line if no intermediate support allowed
    ################################# 
    if not test and sup_max==0:
        indminmulti = 0
        diff = 0.
        while diff <double_max(Csize,LminSpan):
            indminmulti += 1
            diff = Line[indminmulti,0]-Line[0,0]   
        for posi in range(indmax-1,indminmulti-1,-1):
            if test_hfor:
                Hg = Line[posi,7]
            else:
                Hg = Hend            
            test0,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,0,posi,Hg,Htower,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,0,slope_prev=-9999)
            if test0:
                Span[0,0],Span[0,1],Span[0,2],Span[0,3]=D,H,diag,slope
                Span[0,4],Span[0,5],Span[0,6],Span[0,7]=fact,Xup,Zup,Lo
                Span[0,8],Span[0,9],Span[0,10],Span[0,11]=Th,Tv,sqrt(Th*Th+Tv*Tv),sqrt(Th*Th+(Tv-F-Lo*g*q1)**2)
                Span[0,14],Span[0,15]=Hg,posi
                test=1
                break                
    #################################
    # End Cut the line if no intermediate support allowed
    ################################# 
    #################################
    # Start intermediate support position optimisation
    ################################# 
    if not test and sup_max>0: 
        #Get index respecting minimum length between two pylons
        indmaxmulti = indmax
        diff = 0.
        while diff <double_max(Csize,LminSpan) and indmaxmulti>0:
            indmaxmulti -= 1
            diff = Line[indmax,0]-Line[indmaxmulti,0]
        if indmaxmulti==0:
            test=1        
    if not test and sup_max>0:     
        #Generate table to compare results 
        Tab = -9999*np.ones((indmaxmulti*100*nbconfig,15*(sup_max+1)),dtype=np.float) 
        lineTab=0
        Tabis = -9999*np.ones((1,15*(sup_max+1)),dtype=np.float)  
        lineTabis = 0
        # posi1 Tdown1 Dsupdep1 slope1 ....  
        ########################################
        #Optimize int sup implantation
        ########################################        
        test=1
        intsup=1
        newTmax = Tmax
        Dsupdep = 0
        pg=0
        if test_hfor:
            Hginit = Line[0,7]
        else:
            Hginit = Hend
        best=0
        slope_prev=-9999
        indmin=0
        while intsup<=sup_max and not best:
            #Optimize
            for p in range(0,nblineTabis):
                if intsup>1:
                    lineTabis = p
                    col = (intsup-2)*15
                    indmin = ceil(Tabis[p,13+col])  
                    pg=indmin
                    newTmax=Tabis[p,10+col]
                    Dsupdep=0
                    Hginit=Tabis[p,14+col]                
                    for c in range(col+2,1,-15):
                        Dsupdep+=Tabis[p,c]    
                    slope_prev = Tabis[p,3+col]                   
                for posi in range(indmaxmulti,indmin,-1):
                    # Get info from nearest span 
                    Hd = Line[posi,7]
                    Hg=Hginit
                    test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Hd,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                    if test1: 
                        Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                        Tab[lineTab,0:(intsup-1)*15]=Tabis[lineTabis,0:(intsup-1)*15]
                        Tab[lineTab,(intsup-1)*15:((intsup-1)*15+15)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hg,posi,Hd
                        lineTab+=1
                        # Get info from farest span
                        test2,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,posi,indmax,Hd,Htower,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,Tdown,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,diag+Dsupdep,slope)
                        if test2:
                            best=1
                            Tab[lineTab,0:intsup*15]=Tab[lineTab-1,0:intsup*15]                         
                            Tab[lineTab,intsup*15:(intsup*15+15)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hd,indmax,Htower
                            break           
                if best:                    
                    break
            if not best:                
                if lineTab==0:
                    test=0
                    intsup-=2
                    if Tabis[0,0]>0:                    
                        Tab[lineTab] = Tabis[0]
                    else:
                        Tab[lineTab] *= 0
                    break
                Tabis = get_Tabis2(Tab,lineTab,nbconfig,intsup,indmax)
                nblineTabis=Tabis.shape[0]
                if nblineTabis>0:
                    Tab = -9999*np.ones((indmaxmulti*nbconfig*Tabis.shape[0]*100,15*(sup_max+1)),dtype=np.float)  
                    intsup+=1  
                    lineTab=0
                else:
                    test=0
                    intsup=-2
                    break
        ######## cut the line at the farest position if sup_max>0
        if not best and test:            
            lineTab=0
            for p in range(0,nblineTabis):
                lineTabis = p
                col = (intsup-2)*15
                pg = ceil(Tabis[p,13+col]) 
                Hg=Tabis[p,14+col]
                indminmulti = pg
                diff = 0.
                while diff <double_max(Csize,LminSpan):
                    indminmulti += 1
                    diff = Line[indminmulti,0]-Line[pg,0]  
                newTmax=Tabis[p,10+col]
                Dsupdep=0
                for c in range(col+2,1,-15):
                    Dsupdep+=Tabis[p,c]    
                slope_prev = Tabis[p,3+col]
                for posi in range(indmax-1,indminmulti-1,-1):
                    # Get info from nearest span                    
                    test1,D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,F = test_Span(Line,pg,posi,Hg,Htower,Hline_min,Hline_max,slope_min,slope_max,Alts,Fo,newTmax,q1,q2,q3,EAo,rastLosup,rastTh,rastTv,Csize,angle_intsup,Dsupdep,slope_prev)
                    if test1: 
                        Tdown = sqrt(Th*Th+(Tv-q1*g*Lo)*(Tv-q1*g*Lo))#Not take into account F1 cause we want to check other loaded spans
                        Tab[lineTab,0:(intsup-1)*15]=Tabis[lineTabis,0:(intsup-1)*15]
                        Tab[lineTab,(intsup-1)*15:((intsup-1)*15+15)]=D,H,diag,slope,fact,Xup,Zup,Lo,Th,Tv,Tcalc,sqrt(Th*Th+(Tv-F-Lo*g*q1)**2),Hg,posi,Htower
                        lineTab+=1             
            if lineTab==0:
                intsup-=2
                lineTab=0                                   
                Tab[lineTab] = Tabis[0]
            else:
                Tab = get_Tabis2(Tab,lineTab,1,intsup,indmax)                
                lineTab=0       
                intsup=sup_max 
        ######## Save Span carac 
        for i in range(0,intsup+1):
            col = i*15            
            Span[i,[0,1,2,3,4,5,6,7,8,9,10,11,14,15]]=Tab[lineTab,col:col+14]
    return Span  

@cython.cdivision(True)  
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef OptPyl_Down(np.ndarray[dtypef_t,ndim=2] Line,np.ndarray[dtypef_t,ndim=1] Alts,np.ndarray[dtypef_t,ndim=2] Span,
                  double Htower,double Hintsup,double Hend,double q1,double q2,double q3,double Fo,double Hline_min,
                  double Hline_max,double Csize,double angle_intsup,double EAo,double E,double d,
                  unsigned int sup_max,np.ndarray[dtypef_t,ndim=2] rastLosup,np.ndarray[dtypef_t,ndim=2] rastTh,
                  np.ndarray[dtypef_t,ndim=2] rastTv,double Tmax,double LminSpan,double slope_min,
                  double slope_max,double Lmax,unsigned int test_hfor,unsigned int nbconfig=5):
    """
    Cable machine en bas
    Permet de recuper la meilleure configuration avec optimisation de position et hauteur des supports intermediaires
        
    """  
    cdef int indmax = Line.shape[0]-1,i=-1,j=sup_max
    cdef double Dmax=Line[indmax,0]
    cdef int test = 0
    cdef np.ndarray[dtypef_t,ndim=2] Spanbis = np.zeros_like(Span)
    while not test:
        indmax = Line.shape[0]-1 
        if indmax==1:
            test=0
            break
        Span = OptPyl_Up2(Line,Alts,Span*0,Htower,Hintsup,Hend,q1,q2,q3,Fo,Hline_min,Hline_max,Csize,angle_intsup,EAo,E,d,sup_max,rastLosup,rastTh,rastTv,Tmax,LminSpan,slope_min,slope_max,Lmax,test_hfor,nbconfig)
        if max_array_f(Span[:,15])==indmax:
            test=1
            break
        else:
            Line=Line[1:]
    if test:
        for j in range(sup_max,-1,-1):
            if Span[j,0]>0:                
                break
        i=-1
        while j>-1:
            i+=1
            Spanbis[i]=Span[j]  
            Spanbis[i,5]=Dmax-Span[j,5]
            Spanbis[i,15]=indmax-Span[j-1,15]
            j-=1               
        Spanbis[i,15]=indmax
        Spanbis[:,4]*=-1
    return Spanbis

@cython.cdivision(True)  
@cython.boundscheck(False)
@cython.wraparound(False)  
cdef OptPyl_Down_NoH(np.ndarray[dtypef_t,ndim=2] Line,np.ndarray[dtypef_t,ndim=1] Alts,np.ndarray[dtypef_t,ndim=2] Span,
                  double Htower,double Hintsup,double Hend,double q1,double q2,double q3,double Fo,double Hline_min,
                  double Hline_max,double Csize,double angle_intsup,double EAo,double E,double d,
                  unsigned int sup_max,np.ndarray[dtypef_t,ndim=2] rastLosup,np.ndarray[dtypef_t,ndim=2] rastTh,
                  np.ndarray[dtypef_t,ndim=2] rastTv,double Tmax,double LminSpan,double slope_min,
                  double slope_max,double Lmax,unsigned int test_hfor,unsigned int nbconfig=5):
    """
    Cable machine en bas
    Permet de recuper la meilleure configuration avec optimisation de position (mais pas de la hauteur) des supports intermediaires
        
    """  
    cdef int indmax = Line.shape[0]-1,i=-1,j=sup_max
    cdef double Dmax=Line[indmax,0]
    cdef int test = 0
    cdef np.ndarray[dtypef_t,ndim=2] Spanbis = np.zeros_like(Span)
    while not test:
        indmax = Line.shape[0]-1 
        if indmax==1:
            test=0
            break
        Span = OptPyl_Up2_NoH(Line,Alts,Span*0,Htower,Hintsup,Hend,q1,q2,q3,Fo,Hline_min,Hline_max,Csize,angle_intsup,EAo,E,d,sup_max,rastLosup,rastTh,rastTv,Tmax,LminSpan,slope_min,slope_max,Lmax,test_hfor,nbconfig)
        if max_array_f(Span[:,15])==indmax:
            test=1
            break
        else:
            Line=Line[1:]
    if test:
        for j in range(sup_max,-1,-1):
            if Span[j,0]>0:                
                break
        i=-1
        while j>-1:
            i+=1
            Spanbis[i]=Span[j]  
            Spanbis[i,5]=Dmax-Span[j,5]
            Spanbis[i,15]=indmax-Span[j-1,15]
            j-=1               
        Spanbis[i,15]=indmax
        Spanbis[:,4]*=-1
    return Spanbis
              

# Mask pour determiner l'emprise du cable
@cython.boundscheck(False)
@cython.wraparound(False)  
def mask_3(np.ndarray[dtype_t,ndim=2] matrice,int nbpixel_bis):
    cdef unsigned int nline = matrice.shape[0]
    cdef unsigned int ncol = matrice.shape[1]
    cdef unsigned int top = nline
    cdef unsigned int bottom = 0
    cdef unsigned int left = ncol
    cdef unsigned int right = 0
    cdef unsigned int x = 0
    cdef unsigned int y = 0
    for y from 0 <= y < nline:
        for x from 0 <= x < ncol:
            if matrice[y,x]==1:
                if y < top: top=y
                if y > bottom: bottom=y
                if x < left: left=x
                if x > right: right=x
    top = int_max(0,top - nbpixel_bis)
    bottom = int_min(nline,bottom + 1 + nbpixel_bis)
    left = int_max(0,left - nbpixel_bis)
    right = int_min(ncol,right + 1 + nbpixel_bis)
    return top,bottom,left,right

# Renvoie l'emprise reelle du cable
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype_t,ndim=2] Line_emprise(np.ndarray[dtype_t,ndim=2] zone,int direction,unsigned int Buffer_cote,
                 np.ndarray[dtypeu8_t,ndim=2] Ligne_perpendic):
    cdef np.ndarray[dtype_t,ndim=2] zone_ok = zone.copy()
    cdef unsigned int h3 = direction*(1+2*Buffer_cote)
    cdef unsigned int b3 = h3+(1+2*Buffer_cote)
    cdef np.ndarray[dtypeu8_t,ndim=2] Mask = Ligne_perpendic[h3:b3]*1
    cdef unsigned int L = 0
    cdef unsigned int C = 0
    cdef Py_ssize_t nline = zone.shape[0]
    cdef Py_ssize_t ncol = zone.shape[1]
    cdef unsigned int h,b,l,r,buf_h,buf_l,y,x,x1,y1
    for L from 0 <= L <nline:
        for C from 0 <= C <ncol:
            if zone[L,C]==1:
                h = int_max(0,L - Buffer_cote)
                buf_h = 0
                if h == 0: buf_h = Buffer_cote-L
                b = int_min(nline,L + 1 + Buffer_cote)
                l = int_max(0,C - Buffer_cote)  
                buf_l = 0
                if l==0:buf_l = Buffer_cote-C
                r = int_min(ncol,C + 1 + Buffer_cote)
                y = h
                x = l
                y1 = buf_h
                for y from h <= y < b:
                    x1 = buf_l
                    for x from l<= x < r:                        
                        zone_ok[y,x]=int_min((zone_ok[y,x]+Mask[y1,x1]),1)
                        x1 +=1
                    y1 +=1
    return zone_ok

# Sauvegarde les resultats dans un raster entier
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype_t,ndim=2] concatenate_int(np.ndarray[dtype_t,ndim=2] zone,unsigned int h,unsigned int b,
                 unsigned int l,unsigned int r,np.ndarray[dtype_t,ndim=2] Mask):
    cdef unsigned int y = h
    cdef unsigned int x = l
    cdef unsigned int y1 = 0
    cdef unsigned int x1
    for y from h <= y < b:
        x1 = 0
        for x from l<= x < r:
            zone[y,x]=(zone[y,x]+Mask[y1,x1])
            x1 +=1
        y1 +=1
    return zone

# Sauvegarde les resultats dans un raster de nombre flottant
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtypef_t,ndim=2] concatenate_float(np.ndarray[dtypef_t,ndim=2] zone,unsigned int h,unsigned int b,
                 unsigned int l,unsigned int r,np.ndarray[dtypef_t,ndim=2] Mask):
    cdef unsigned int y = h
    cdef unsigned int x = l
    cdef unsigned int y1 = 0
    cdef unsigned int x1
    for y from h <= y < b:
        x1 = 0
        for x from l<= x < r:
            zone[y,x]=(zone[y,x]+Mask[y1,x1])
            x1 +=1
        y1 +=1
    return zone
    
##############################################################################################################################################
### Fonctions pour le modele tracteur
##############################################################################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef skid_debusq_RF(np.ndarray[dtype_t,ndim=2] Lien_RF,np.ndarray[dtypef32_t,ndim=2] MNT,
                     np.ndarray[dtype16_t,ndim=2] Row_line,np.ndarray[dtype16_t,ndim=2] Col_line,
                     np.ndarray[dtypef_t,ndim=2] D_line,np.ndarray[dtype16_t,ndim=1] Nbpix_line,
                     double coeff, double orig, double Pmax_up, double Pmax_down,int damont,
                     int daval,double Csize, int nrows,int ncols,np.ndarray[dtype8_t,ndim=2] Zone_ok):
    cdef int Max_distance = 100000,az,nbpix,Y,X,i,j,test,testRF
    cdef double dmin = int_min(damont,daval)
    cdef double dmax
    cdef np.ndarray[dtype32_t,ndim=2] Out_distance = np.ones((nrows,ncols),dtype=np.int32)*(Max_distance+1)
    cdef np.ndarray[dtype32_t,ndim=2] L_des = np.zeros((nrows,ncols),dtype=np.int32)
    cdef unsigned int coordX,coordY,nb_pixel_RF = Lien_RF.shape[0],pixel
    cdef double Hdist,Alt_RF,Alt_pixel,coef_l,Hline,dist
    for pixel from 1<=pixel<nb_pixel_RF:
        coordY = Lien_RF[pixel,0]
        coordX = Lien_RF[pixel,1]
        Alt_RF = MNT[coordY,coordX]
        testRF=0
        for az from 0<=az<360:
            nbpix = Nbpix_line[az]
            for i from 1<=i<nbpix:
                Y=coordY+Row_line[az,i]
                if Y<0:break
                if Y>=nrows:break               
                X=coordX+Col_line[az,i]
                if X<0:break
                if X>=ncols:break              
                if not Zone_ok[Y,X]:break
                #get pixel info
                Hdist = D_line[az,i]
                Alt_pixel = MNT[Y,X]                
                dist = sqrt(Hdist*Hdist+(Alt_pixel-Alt_RF)*(Alt_pixel-Alt_RF))
                coef_l = (Alt_pixel-Alt_RF)/(Hdist)
                if Out_distance[Y,X] > dist:
                    #check if straight line is OK before
                    j=1
                    test=1
                    for j from 1<=j<i:
                        Hline = (Alt_RF+coef_l*D_line[az,j]+10.)-MNT[coordY+Row_line[az,j],coordX+Col_line[az,j]]
                        if Hline<0 or Hline>30.:
                            test=0
                            break                                      
                    if not test:break                
                    if dist> dmin:                        
                        if coef_l <= Pmax_down: dmax=daval
                        elif coef_l > Pmax_up:dmax=damont
                        else: dmax=orig/(1-coeff*coef_l/sqrt(1+coef_l*coef_l))
                        if dist>dmax:break                
                    Out_distance[Y,X]=int(dist+0.5)
                    L_des[Y,X]=pixel 
                    testRF=1

        if testRF:
            Out_distance[coordY,coordX]=0
            L_des[coordY,coordX]=pixel                    
    for Y in range(0,nrows,1):
        for X in range(0,ncols,1):
            if Out_distance[Y,X]>Max_distance:
                Out_distance[Y,X]=-9999
                L_des[Y,X]=-9999 
    return Out_distance,L_des

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef skid_debusq_Piste(np.ndarray[dtype_t,ndim=2] Lien_RF,np.ndarray[dtypef32_t,ndim=2] MNT,
                        np.ndarray[dtype16_t,ndim=2] Row_line,np.ndarray[dtype16_t,ndim=2] Col_line,
                        np.ndarray[dtypef_t,ndim=2] D_line,np.ndarray[dtype16_t,ndim=1] Nbpix_line,
                        double coeff, double orig, double Pmax_up, double Pmax_down,unsigned int damont,
                        unsigned int daval,double Csize, int nrows,int ncols,np.ndarray[dtype8_t,ndim=2] Zone_ok):
    cdef int Max_distance = 100000,az,nbpix,Y,X,i,j,test,testRF
    cdef double dmin = int_min(damont,daval)
    cdef double dmax
    cdef np.ndarray[dtype32_t,ndim=2] Out_distance = np.ones((nrows,ncols),dtype=np.int32)*(Max_distance+1)
    cdef np.ndarray[dtype32_t,ndim=2] L_des = np.zeros((nrows,ncols),dtype=np.int32)
    cdef np.ndarray[dtype32_t,ndim=2] Dpis = np.zeros((nrows,ncols),dtype=np.int32)
    cdef unsigned int coordX,coordY,nb_pixel_RF = Lien_RF.shape[0],pixel =1
    cdef double Hdist,Alt_RF,Alt_pixel,coef_l,Hline,dist,dpist
    for pixel from 1<=pixel<nb_pixel_RF:
        coordY = Lien_RF[pixel,0]
        coordX = Lien_RF[pixel,1]
        Alt_RF = MNT[coordY,coordX] 
        testRF=0
        dpist = Lien_RF[pixel,2]
        for az from 0<=az<360:
            nbpix = Nbpix_line[az]            
            for i from 1<=i<nbpix:
                Y=coordY+Row_line[az,i]
                if Y<0:break
                if Y>=nrows:break               
                X=coordX+Col_line[az,i]
                if X<0:break
                if X>=ncols:break              
                if not Zone_ok[Y,X]:break
                #get pixel info
                Hdist = D_line[az,i]
                Alt_pixel = MNT[Y,X]
                dist = sqrt(Hdist*Hdist+(Alt_pixel-Alt_RF)*(Alt_pixel-Alt_RF))
                coef_l = (Alt_pixel-Alt_RF)/(Hdist)
                if Out_distance[Y,X]>dist:
                #check if straight line is OK before
                    j=1
                    test=1
                    for j from 1<=j<i:
                        Hline = (Alt_RF+coef_l*D_line[az,j]+10.)-MNT[coordY+Row_line[az,j],coordX+Col_line[az,j]]
                        if Hline<0 or Hline>30.:
                            test=0
                            break                                      
                    if not test:break                
                    if dist> dmin:                        
                        if coef_l <= Pmax_down: dmax=daval
                        elif coef_l > Pmax_up:dmax=damont
                        else: dmax=orig/(1-coeff*coef_l/sqrt(1+coef_l*coef_l))
                        if dist>dmax:break                          
                    Out_distance[Y,X]=int(dist+0.5)
                    L_des[Y,X]=pixel 
                    Dpis[Y,X]=int(dpist+0.5)
                    testRF=1
        if testRF:
            Out_distance[coordY,coordX]=0
            L_des[coordY,coordX]=pixel 
            Dpis[coordY,coordX]=int(dpist+0.5)
    for Y in range(0,nrows,1):
        for X in range(0,ncols,1):
            if Out_distance[Y,X]>Max_distance:
                Out_distance[Y,X]=-9999
                L_des[Y,X]=-9999 
                Dpis[Y,X]=-9999 
    return Out_distance,L_des,Dpis


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef skid_debusq_contour(np.ndarray[dtype_t,ndim=2] Lien_RF,np.ndarray[dtypef32_t,ndim=2] MNT,
                          np.ndarray[dtype16_t,ndim=2] Row_line,np.ndarray[dtype16_t,ndim=2] Col_line,
                          np.ndarray[dtypef_t,ndim=2] D_line,np.ndarray[dtype16_t,ndim=1] Nbpix_line,
                          double coeff, double orig, double Pmax_up, double Pmax_down,unsigned int damont,
                          unsigned int daval,double Csize, int nrows,int ncols,np.ndarray[dtype8_t,ndim=2] Zone_ok):
    cdef int Max_distance = 100000,az,nbpix,Y,X,i,j,test,testRF,lienRF,lienPiste
    cdef double dmin = int_min(damont,daval)
    cdef double dmax
    cdef np.ndarray[dtype32_t,ndim=2] Out_distance = np.ones((nrows,ncols),dtype=np.int32)*(Max_distance+1)
    cdef np.ndarray[dtype32_t,ndim=2] L_RF = np.zeros((nrows,ncols),dtype=np.int32)
    cdef np.ndarray[dtype32_t,ndim=2] L_Piste = np.zeros((nrows,ncols),dtype=np.int32)
    cdef np.ndarray[dtype32_t,ndim=2] Dpis = np.zeros((nrows,ncols),dtype=np.int32)
    cdef np.ndarray[dtype32_t,ndim=2] Dfor = np.zeros((nrows,ncols),dtype=np.int32)
    cdef unsigned int coordX,coordY,nb_pixel_RF = Lien_RF.shape[0],pixel =1
    cdef double Hdist,Alt_RF,Alt_pixel,coef_l,Hline,dist,dpist,dfor
    for pixel from 1<=pixel<nb_pixel_RF:
        coordY = Lien_RF[pixel,0]
        coordX = Lien_RF[pixel,1]
        Alt_RF = MNT[coordY,coordX] 
        testRF=0
        lienRF=Lien_RF[pixel,2]
        lienPiste=Lien_RF[pixel,3]
        dpist = Lien_RF[pixel,5]
        dfor = Lien_RF[pixel,4]        
        for az from 0<=az<360:
            nbpix = Nbpix_line[az]            
            for i from 1<=i<nbpix:
                Y=coordY+Row_line[az,i]
                if Y<0:break
                if Y>=nrows:break               
                X=coordX+Col_line[az,i]
                if X<0:break
                if X>=ncols:break              
                if not Zone_ok[Y,X]:break
                #get pixel info
                Hdist = D_line[az,i]
                Alt_pixel = MNT[Y,X]            
                coef_l = (Alt_pixel-Alt_RF)/(Hdist)
                dist = sqrt(Hdist*Hdist+(Alt_pixel-Alt_RF)*(Alt_pixel-Alt_RF))
                if (Out_distance[Y,X]+Dpis[Y,X]+Dfor[Y,X])>(dist+dpist+dfor):
                    j=1
                    test=1
                    for j from 1<=j<i:
                        Hline = (Alt_RF+coef_l*D_line[az,j]+10.)-MNT[coordY+Row_line[az,j],coordX+Col_line[az,j]]
                        if Hline<0 or Hline>30.:
                            test=0
                            break                                      
                    if not test:break                
                    if dist> dmin:                        
                        if coef_l <= Pmax_down: dmax=daval
                        elif coef_l > Pmax_up:dmax=damont
                        else: dmax=orig/(1-coeff*coef_l/sqrt(1+coef_l*coef_l))
                        if dist>dmax:break                          
                    Out_distance[Y,X]=int(dist+0.5)
                    L_RF[Y,X]=lienRF 
                    L_Piste[Y,X]=lienPiste
                    Dpis[Y,X]=int(dpist+0.5)
                    Dfor[Y,X]=int(dfor+0.5)
                    testRF=1
        if testRF:
            Out_distance[coordY,coordX]=0
            L_RF[coordY,coordX]=lienRF 
            L_Piste[coordY,coordX]=lienPiste
            Dpis[coordY,coordX]=int(dpist+0.5)
            Dfor[coordY,coordX]=int(dfor+0.5)
    for Y in range(0,nrows,1):
        for X in range(0,ncols,1):
            if Out_distance[Y,X]>Max_distance:
                Out_distance[Y,X]=-9999
                L_RF[Y,X]=-9999
                L_Piste[Y,X]=-9999
                Dpis[Y,X]=-9999
                Dfor[Y,X]=-9999
    return Out_distance,L_RF,L_Piste,Dpis,Dfor

##############################################################################################################################################
### Fonctions pour le modele porteur
##############################################################################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned int seek_ind(np.ndarray[dtypef32_t,ndim=2] Tab, unsigned int x, unsigned y):
    cdef unsigned int nline = Tab.shape[0]
    cdef unsigned int y1=1,ind=1
    while y1<nline:
        if Tab[y1,0]==y:
            if Tab[y1,1]==x:
                ind=y1
                break
        y1+=1
    return ind

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned int seek_ind_i(np.ndarray[dtype32_t,ndim=2] Tab, unsigned int x, unsigned y):
    cdef unsigned int nline = Tab.shape[0]
    cdef unsigned int y1=1,ind=1
    while y1<nline:
        if Tab[y1,0]==y:
            if Tab[y1,1]==x:
                ind=y1
                break
        y1+=1
    return ind

# Link between forest road and public network
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype_t,ndim=2] Link_RF_res_pub(np.ndarray[dtype32_t,ndim=2] Tab_res_pub,
                                                 np.ndarray[dtypef32_t,ndim=2] cost_rast,
                                                 np.ndarray[dtype8_t,ndim=2] RF, 
                                                 np.ndarray[dtype8_t,ndim=2] Res_pub, 
                                                 np.ndarray[dtypef32_t,ndim=2] Link_RF,
                                                 double Csize,
                                                 unsigned int Max_distance=100000):
    cdef unsigned int nline = cost_rast.shape[0]
    cdef unsigned int ncol = cost_rast.shape[1]
    cdef double diag = 1.414214*Csize
    cdef double direct = Csize,dist_ac = Csize
    # Creation des rasters de sorties
    cdef unsigned int x,y,x1,y1,test
    cdef double Dist
    cdef unsigned int nb_pixel_res_pub = Tab_res_pub.shape[0]
    cdef unsigned int nb_pixel_RF = Link_RF.shape[0]
    cdef unsigned int pixel =1,ind
    # Initialisation du raster
    while pixel <  nb_pixel_res_pub:
        y1 = Tab_res_pub[pixel,0]
        x1 = Tab_res_pub[pixel,1] 
        if RF[y1,x1]==1:
            ind = seek_ind(Link_RF, x1, y1)
            Link_RF[ind,2] = 0                             
            Link_RF[ind,3] = pixel
            Link_RF[ind,4] = 1        
        pixel+=1
    # Traitement complet    
    while dist_ac<=Max_distance:
        test = 0
        pixel=1
        while pixel <  nb_pixel_RF:
            if Link_RF[pixel,4]==1:
                test=1
                y1 = int(Link_RF[pixel,0])
                x1 = int(Link_RF[pixel,1])
                Link_RF[pixel,4] = 2
                dist_ac = Link_RF[pixel,2] 
                for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                    for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                        if RF[y,x]==1:
                            if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                            else: Dist = cost_rast[y,x]*direct+dist_ac
                            ind = seek_ind(Link_RF, x, y)
                            if Link_RF[ind,2]>Dist:
                                Link_RF[ind,2] = Dist                               
                                Link_RF[ind,3] = Link_RF[pixel,3] 
                                Link_RF[ind,4] = 1
            pixel+=1
        if test==0:
            break
    # Verifie si certaines routes ne sont pas connectee    
    test=0
    pixel=1
    # Initialisation du raster
    while pixel <  nb_pixel_RF:
        if Link_RF[pixel,2]==100001:
            y1 = int(Link_RF[pixel,0])
            x1 = int(Link_RF[pixel,1])
            for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):                        
                    if Res_pub[y,x]==1:
                        if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag
                        else: Dist = cost_rast[y,x]*direct 
                        ind = seek_ind_i(Tab_res_pub, x, y)
                        Link_RF[pixel,2] = Dist                             
                        Link_RF[pixel,3] = ind
                        Link_RF[pixel,4] = 1
                        test=1
        pixel+=1
    # Traitement complet    
    while test:
        test = 0
        pixel=1
        while pixel <  nb_pixel_RF:
            if Link_RF[pixel,4]==1:
                test=1
                y1 = int(Link_RF[pixel,0])
                x1 = int(Link_RF[pixel,1])
                Link_RF[pixel,4] = 2
                dist_ac = Link_RF[pixel,2] 
                for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                    for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                        if RF[y,x]==1:
                            if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                            else: Dist = cost_rast[y,x]*direct+dist_ac
                            ind = seek_ind(Link_RF, x, y)
                            if Link_RF[ind,2]>Dist:
                                Link_RF[ind,2] = Dist                               
                                Link_RF[ind,3] = Link_RF[pixel,3] 
                                Link_RF[ind,4] = 1
            pixel+=1
    return Link_RF[:,0:-1]

# Link between forest tracks and RF and public network
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype_t,ndim=2] Link_tracks_res_pub(np.ndarray[dtype32_t,ndim=2] Tab_res_pub,
                                                     np.ndarray[dtype32_t,ndim=2] Link_RF,
                                                     np.ndarray[dtypef32_t,ndim=2] cost_rast,
                                                     np.ndarray[dtype8_t,ndim=2] Piste,
                                                     np.ndarray[dtype8_t,ndim=2] RF,
                                                     np.ndarray[dtype8_t,ndim=2] Res_Pub,
                                                     np.ndarray[dtypef32_t,ndim=2] Link_Piste,
                                                     double Csize,unsigned int Max_distance=100000):
    cdef unsigned int nline = cost_rast.shape[0]
    cdef unsigned int ncol = cost_rast.shape[1]
    cdef float diag = 1.414214*Csize
    cdef float direct = Csize
    cdef unsigned int x,y,x1,y1,test
    cdef float Dist,dist_ac = Csize
    cdef unsigned int nb_pixel_res_pub = Tab_res_pub.shape[0]
    cdef unsigned int nb_pixel_RF = Link_RF.shape[0]
    cdef unsigned int nb_pixel_Piste = Link_Piste.shape[0]
    cdef unsigned int pixel =1,ind
    # Initialisation du raster depuis reseau public
    while pixel <  nb_pixel_res_pub:
        y1 = Tab_res_pub[pixel,0]
        x1 = Tab_res_pub[pixel,1] 
        if Piste[y1,x1]==1:
            ind = seek_ind(Link_Piste, x1, y1)
            Link_Piste[ind,2]= 0                             
            Link_Piste[ind,3]= 0
            Link_Piste[ind,4]= -9999
            Link_Piste[ind,5]= pixel
            Link_Piste[ind,6]= 1                        
        pixel+=1
    pixel=1
    # Initialisation du raster depuis route_for
    while pixel < nb_pixel_RF:
        y1 = Link_RF[pixel,0]
        x1 = Link_RF[pixel,1] 
        if Piste[y1,x1]==1:
            ind = seek_ind(Link_Piste, x1, y1)
            Link_Piste[ind,2]= 0                            
            Link_Piste[ind,3]= Link_RF[pixel,2]
            Link_Piste[ind,4]= pixel
            Link_Piste[ind,5]= Link_RF[pixel,3]
            Link_Piste[ind,6]= 1
        pixel+=1
    # Traitement complet    
    while dist_ac<=Max_distance:
        test = 0
        pixel=1
        while pixel <  nb_pixel_Piste:
            if Link_Piste[pixel,6]==1:
                test=1
                y1 = int(Link_Piste[pixel,0])
                x1 = int(Link_Piste[pixel,1])
                Link_Piste[pixel,6]=2
                dist_ac=Link_Piste[pixel,2] 
                for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                    for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                        if Piste[y,x]==1:
                            if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                            else: Dist = cost_rast[y,x]*direct+dist_ac
                            ind = seek_ind(Link_Piste, x, y)
                            if Link_Piste[ind,2]>Dist:
                                Link_Piste[ind,2] = Dist                              
                                Link_Piste[ind,3]= Link_Piste[pixel,3]
                                Link_Piste[ind,4]= Link_Piste[pixel,4]
                                Link_Piste[ind,5]= Link_Piste[pixel,5]
                                Link_Piste[ind,6]= 1
            pixel+=1
        if test==0:
            break
    #Verifie si certaines pistes ne sont pas connectee    
    test=0
    pixel=1
    # Initialisation du raster
    while pixel <  nb_pixel_Piste:
        if Link_Piste[pixel,2]==100001:
            y1 = int(Link_Piste[pixel,0])
            x1 = int(Link_Piste[pixel,1])
            for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):                        
                    if Res_Pub[y,x]==1:                        
                        if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag
                        else: Dist = cost_rast[y,x]*direct 
                        if Link_Piste[pixel,2]>Dist:
                            ind = seek_ind_i(Tab_res_pub, x, y)
                            Link_Piste[pixel,2]= Dist                             
                            Link_Piste[pixel,3]= 0
                            Link_Piste[pixel,4]= -9999
                            Link_Piste[pixel,5]= ind
                            Link_Piste[pixel,6]= 1
                            test=1
                    if RF[y,x]==1:
                        if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag
                        else: Dist = cost_rast[y,x]*direct 
                        if Link_Piste[pixel,2]>Dist:
                            ind = seek_ind_i(Link_RF, x, y)
                            Link_Piste[pixel,2]= Dist                             
                            Link_Piste[pixel,3]= Link_RF[ind,2]
                            Link_Piste[pixel,4]= ind
                            Link_Piste[pixel,5]= Link_RF[ind,3]
                            Link_Piste[pixel,6]= 1
                            test=1                        
        pixel+=1  
    #traitement complet
    while test:
        test = 0
        pixel=1
        while pixel <  nb_pixel_Piste:
            if Link_Piste[pixel,6]==1:
                test=1
                y1 = int(Link_Piste[pixel,0])
                x1 = int(Link_Piste[pixel,1])
                Link_Piste[pixel,6]=2
                dist_ac=Link_Piste[pixel,2] 
                for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                    for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                        if Piste[y,x]==1:
                            if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                            else: Dist = cost_rast[y,x]*direct+dist_ac
                            ind = seek_ind(Link_Piste, x, y)
                            if Link_Piste[ind,2]>Dist:
                                Link_Piste[ind,2] = Dist                              
                                Link_Piste[ind,3]= Link_Piste[pixel,3]
                                Link_Piste[ind,4]= Link_Piste[pixel,4]
                                Link_Piste[ind,5]= Link_Piste[pixel,5]
                                Link_Piste[ind,6]= 1
            pixel+=1
    return Link_Piste[:,0:-1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Dfwd_flat_forest_road(np.ndarray[dtype_t,ndim=2] Link_RF, np.ndarray[dtypef32_t,ndim=2] cost_rast,
                            np.ndarray[dtype8_t,ndim=2] zone_rast, double Csize, unsigned int Max_distance = 100000):    
    """
    Calcule la distance de trainage  de plus faible faible coût depuis le peuplement jusqu'a la desserte
    """
    cdef unsigned int nline = zone_rast.shape[0]
    cdef unsigned int ncol = zone_rast.shape[1]
    cdef double diag = 1.414214*Csize
    cdef double direct = Csize
    cdef unsigned int nb_pixel_RF = Link_RF.shape[0]
    cdef unsigned int pixel=1,ind,nb_pixel   
    # Creation des rasters de sorties
    cdef np.ndarray[dtype32_t,ndim=2] Out_distance = np.ones_like(zone_rast,dtype=np.int32)*(Max_distance+1)
    cdef np.ndarray[dtype32_t,ndim=2] L_forRF = np.ones_like(zone_rast,dtype=np.int32)*-9999
    cdef unsigned int x,y,x1,y1,test,count_sans_match = 0
    cdef double Dist=0,dist_ac = Csize
    cdef unsigned int h=nline,b=0,l=ncol,r=0
    # Initialisation du raster
    pixel=1
    while pixel < nb_pixel_RF:
        y1 = Link_RF[pixel,0]
        x1 = Link_RF[pixel,1]
        Out_distance[y1,x1] = 0
        L_forRF[y1,x1] = pixel
        for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
            for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):                      
                if zone_rast[y,x]==1:
                    if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag
                    else: Dist = cost_rast[y,x]*direct
                    if Out_distance[y,x] > Dist:
                        Out_distance[y,x] = int(Dist+0.5)                                
                        L_forRF[y,x] = pixel
                        if y<h:h=y
                        if x<l:l=x
                        if y>b:b=y
                        if x>r:r=x
        pixel+=1
    # Traitement complet
    while dist_ac<=Max_distance and count_sans_match <15*Csize:
        y1,x1=h,l
        test = 0
        for y1 from h <= y1 <b:
            for x1 from l <= x1 <r:
                if Out_distance[y1,x1]==dist_ac:
                    test=1
                    for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                        for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                            if zone_rast[y,x]==1: 
                                if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                                else: Dist = cost_rast[y,x]*direct+dist_ac                                 
                                if Out_distance[y,x] > Dist:
                                    Out_distance[y,x] = int(Dist+0.5)
                                    L_forRF[y,x] = L_forRF[y1,x1]
        if test==1:count_sans_match = 0
        else:count_sans_match +=1
        dist_ac +=1
        h = int_max(0,h-1)
        b = int_min(nline,b+1)
        l = int_max(0,l-1)
        r = int_min(ncol,r+1)
    for y in range(0,nline,1):
        for x in range(0,ncol,1):
            if Out_distance[y,x]>Max_distance:
                Out_distance[y,x]=-9999
                L_forRF[y,x] = -9999
    return Out_distance,L_forRF

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Dfwd_flat_forest_tracks(np.ndarray[dtype_t,ndim=2] Link_Piste, np.ndarray[dtypef32_t,ndim=2] cost_rast,
                              np.ndarray[dtype8_t,ndim=2] zone_rast, double Csize, unsigned int Max_distance = 100000):    
    """
    Calcule la distance de trainage  de plus faible faible coût depuis le peuplement jusqu'a la desserte
    """
    cdef unsigned int nline = zone_rast.shape[0]
    cdef unsigned int ncol = zone_rast.shape[1]
    cdef double diag = 1.414214*Csize
    cdef double direct = Csize
    cdef unsigned int nb_pixel_Piste = Link_Piste.shape[0]
    cdef unsigned int pixel=1,ind,nb_pixel   
    # Creation des rasters de sorties
    cdef np.ndarray[dtype32_t,ndim=2] Out_distance = np.ones_like(zone_rast,dtype=np.int32)*(Max_distance+1)    
    cdef np.ndarray[dtype32_t,ndim=2] L_forPiste = np.ones_like(zone_rast,dtype=np.int32)*-9999
    cdef np.ndarray[dtype32_t,ndim=2] Dpiste = np.ones_like(zone_rast,dtype=np.int32)*-9999
    cdef unsigned int x,y,x1,y1,test,count_sans_match = 0
    cdef double Dist=0,dist_ac = Csize
    cdef unsigned int h=nline,b=0,l=ncol,r=0,test1=0
    # Initialisation du raster
    pixel=1
    while pixel < nb_pixel_Piste:
        y1 = Link_Piste[pixel,0]
        x1 = Link_Piste[pixel,1]
        Out_distance[y1,x1] = 0
        L_forPiste[y1,x1] = pixel
        Dpiste[y1,x1] = Link_Piste[pixel,2]        
        for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
            for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):                      
                if zone_rast[y,x]==1:
                    if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag
                    else: Dist = cost_rast[y,x]*direct
                    if Out_distance[y,x]>=Dist and Dpiste[y,x]+2*Out_distance[y,x]>2*Dist+Link_Piste[pixel,2]:
                        Out_distance[y,x] = int(Dist+0.5)                                
                        L_forPiste[y,x] = pixel
                        Dpiste[y,x] = Link_Piste[pixel,2]                        
                        if y<h:h=y
                        if x<l:l=x
                        if y>b:b=y
                        if x>r:r=x
        pixel+=1
    # Traitement complet
    while dist_ac<=Max_distance and count_sans_match <15*Csize:
        y1,x1=h,l
        test = 0
        for y1 from h <= y1 <b:
            for x1 from l <= x1 <r:
                if Out_distance[y1,x1]==dist_ac:
                    test=1
                    for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                        for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                            if zone_rast[y,x]==1: 
                                if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                                else: Dist = cost_rast[y,x]*direct+dist_ac  
                                if Out_distance[y,x]>=Dist and Dpiste[y,x]+2*Out_distance[y,x]>2*Dist+Dpiste[y1,x1]: 
                                    Out_distance[y,x] = int(Dist+0.5)                                
                                    L_forPiste[y,x] = L_forPiste[y1,x1]
                                    Dpiste[y,x] = Dpiste[y1,x1]                                    
        if test==1:count_sans_match = 0
        else:count_sans_match +=1
        dist_ac +=1
        h = int_max(0,h-1)
        b = int_min(nline,b+1)
        l = int_max(0,l-1)
        r = int_min(ncol,r+1)
    for y in range(0,nline,1):
        for x in range(0,ncol,1):
            if Out_distance[y,x]>Max_distance:
                Out_distance[y,x]=-9999
                L_forPiste[y,x] = -9999
                Dpiste[y,x] =-9999
    return Out_distance,L_forPiste,Dpiste 



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef fwd_azimuts_contour(np.ndarray[dtype_t,ndim=2] Lien_RF,np.ndarray[dtypef32_t,ndim=2] MNT,
                          np.ndarray[dtype16_t,ndim=2] Aspect,np.ndarray[dtypef32_t,ndim=2] Pente,
                          np.ndarray[dtype16_t,ndim=2] Row_line,np.ndarray[dtype16_t,ndim=2] Col_line,
                          np.ndarray[dtypef_t,ndim=2] D_line,np.ndarray[dtype16_t,ndim=1] Nbpix_line,
                          double Fwd_max_up, double Fwd_max_down,double Fwd_max_inc,double Forw_Lmax,
                          double Csize, int nrows,int ncols,np.ndarray[dtype8_t,ndim=2] Zone_ok):
    cdef int Max_distance = 100000,az,nbpix,Y,X,i,testRF,lienRF,lienPiste
    cdef np.ndarray[dtype_t,ndim=2] Out_distance = np.ones((nrows,ncols),dtype=np.int32)*(Max_distance+1)
    cdef np.ndarray[dtype_t,ndim=2] Dfor = np.ones((nrows,ncols),dtype=np.int32)*(Max_distance+1)
    cdef np.ndarray[dtype_t,ndim=2] Dpis = np.ones((nrows,ncols),dtype=np.int32)*(Max_distance+1)
    cdef np.ndarray[dtype_t,ndim=2] L_pis = np.zeros((nrows,ncols),dtype=np.int32)*-9999
    cdef np.ndarray[dtype_t,ndim=2] L_RF = np.zeros((nrows,ncols),dtype=np.int32)*-9999
    cdef unsigned int coordX,coordY,nb_pixel_RF = Lien_RF.shape[0],pixel =1
    cdef double Hdist,Alt_RF,Alt_pixel,dist,dpist,dfor,dist_init
    cdef double dif_angle,max_slope
    for pixel from 1<=pixel<nb_pixel_RF:
        coordY = Lien_RF[pixel,0]
        coordX = Lien_RF[pixel,1]
        Alt_RF = MNT[coordY,coordX] 
        testRF=0
        lienRF=Lien_RF[pixel,4]
        lienPiste=Lien_RF[pixel,5]
        dpist = Lien_RF[pixel,2]
        dfor = Lien_RF[pixel,3]     
        dist_init = dpist+dfor        
        for az from 0<=az<360:
            nbpix = Nbpix_line[az]            
            for i from 1<=i<nbpix:
                Y=coordY+Row_line[az,i]
                if Y<0:break
                if Y>=nrows:break               
                X=coordX+Col_line[az,i]
                if X<0:break
                if X>=ncols:break              
                if not Zone_ok[Y,X]:break
                #get pixel info     
                Hdist= D_line[az,i]
                Alt_pixel = MNT[Y,X]  
                dist = sqrt(Hdist*Hdist+(Alt_pixel-Alt_RF)*(Alt_pixel-Alt_RF))
                if dist>Forw_Lmax :break
                #Check slope              
                if MNT[Y,X]>Alt_RF:                                    
                    if Pente[Y,X]>Fwd_max_down:break
                else:
                    if Pente[Y,X]>Fwd_max_up: break
                #Check forwarder inclination   
                dif_angle = (az-Aspect[Y,X])%180.
                max_slope =  fabs(Fwd_max_inc/cos((90.-dif_angle)/180*pi))
                if Pente[Y,X]>max_slope:break 
                #Check forwarder inclination 
                if Dpis[Y,X]==(Max_distance+1):
                    Out_distance[Y,X] = int(dist+0.5)                    
                    Dpis[Y,X] = int(dpist+0.5)
                    Dfor[Y,X] = int(dfor+0.5)
                    L_RF[Y,X] = lienRF
                    L_pis[Y,X] = lienPiste
                    testRF=1                   
                else:
                    if (Out_distance[Y,X]+Dpis[Y,X]*0.1+Dfor[Y,X]) > (dist+dfor+0.1*dpist): 
                        Out_distance[Y,X] = int(dist+0.5)                    
                        Dpis[Y,X] = int(dpist+0.5)
                        Dfor[Y,X] = int(dfor+0.5)
                        L_RF[Y,X] = lienRF
                        L_pis[Y,X] = lienPiste
                        testRF=1 
                                   
        if testRF:
            Out_distance[coordY,coordX] = 0            
            Dpis[coordY,coordX] = int(dpist+0.5)
            Dfor[coordY,coordX] = int(dfor+0.5)
            L_RF[coordY,coordX] = lienRF
            L_pis[coordY,coordX] = lienPiste            
    for Y in range(0,nrows,1):
        for X in range(0,ncols,1):
            if Out_distance[Y,X]>Max_distance:
                Out_distance[Y,X]=-9999
                L_RF[Y,X]=-9999
                L_pis[Y,X]=-9999
                Dpis[Y,X]=-9999
                Dfor[Y,X]=-9999
    return Out_distance,L_RF,L_pis,Dpis,Dfor

# Fonction permettant d'ajouter la portee du bras du porteur
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Fwd_add_contour(np.ndarray[dtype_t,ndim=2] Lien_contour, np.ndarray[dtypef32_t,ndim=2] cost_rast,np.ndarray[dtype8_t,ndim=2] zone_rast,
                      double Forw_portee, double Csize):    
    """
    Calcule la distance de debusquage de plus faible cout depuis les contours de la zone accessible
    """
    cdef unsigned int nline = zone_rast.shape[0]
    cdef unsigned int ncol = zone_rast.shape[1]
    cdef double diag = 1.414214*Csize
    cdef double direct = Csize,Dist,dist_ac = Csize
    cdef unsigned int Max_distance = int(Forw_portee+0.5)
    cdef unsigned int nb_pixel_contour = Lien_contour.shape[0]
    cdef unsigned int h=nline,b=0,l=ncol,r=0
    # Creation des rasters de sorties
    cdef np.ndarray[dtype_t,ndim=2] Out_distance = np.ones_like(zone_rast,dtype=np.int32)*(100+1)  
    cdef np.ndarray[dtype_t,ndim=2] Lien_RF = np.ones_like(zone_rast,dtype=np.int32)*-9999
    cdef np.ndarray[dtype_t,ndim=2] Lien_piste = np.ones_like(zone_rast,dtype=np.int32)*-9999
    cdef np.ndarray[dtype_t,ndim=2] Dpiste = np.ones_like(zone_rast,dtype=np.int32)*-9999
    cdef np.ndarray[dtype_t,ndim=2] Dforet = np.ones_like(zone_rast,dtype=np.int32)*-9999
    cdef unsigned int x,y,x1=l,y1=h,test,count_sans_match = 0,pixel
    # Initialisation du raster
    pixel=1
    while pixel < nb_pixel_contour:
        y1 = Lien_contour[pixel,0]
        x1 = Lien_contour[pixel,1]
        Out_distance[y1,x1] = 0  
        Lien_RF[y1,x1] = Lien_contour[pixel,3]
        Lien_piste[y1,x1] = Lien_contour[pixel,5]
        Dpiste[y1,x1] = Lien_contour[pixel,4]
        Dforet[y1,x1] = Lien_contour[pixel,2]
        for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
            for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):                        
                if zone_rast[y,x]==1:
                    if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag
                    else: Dist = cost_rast[y,x]*direct
                    if Dist <= Max_distance:
                        if Out_distance[y,x]==101:
                            Out_distance[y,x] = int(Dist+0.5)                                
                            Lien_RF[y,x] = Lien_RF[y1,x1]
                            Lien_piste[y,x] = Lien_piste[y1,x1]
                            Dpiste[y,x] = Dpiste[y1,x1]
                            Dforet[y,x] = Dforet[y1,x1]
                            if y<h:h=y
                            if x<l:l=x
                            if y>b:b=y
                            if x>r:r=x
                        else:
                            if Out_distance[y,x]+0.1*Dpiste[y,x]+Dforet[y,x]>Dist+0.1*Dpiste[y1,x1]+Dforet[y1,x1]:
                                Out_distance[y,x] = int(Dist+0.5)                                
                                Lien_RF[y,x] = Lien_RF[y1,x1]
                                Lien_piste[y,x] = Lien_piste[y1,x1]
                                Dpiste[y,x] = Dpiste[y1,x1]
                                Dforet[y,x] = Dforet[y1,x1]
                                if y<h:h=y
                                if x<l:l=x
                                if y>b:b=y
                                if x>r:r=x                        
        pixel+=1
    # Traitement complet
    while dist_ac<=100 and count_sans_match <15*Csize:
        test = 0
        y1,x1=h,l
        for y1 from h <= y1 <b:
            for x1 from l <= x1 <r:
                if Out_distance[y1,x1]==dist_ac:
                    test=1   
                    for y in range(int_max(0,y1-1),int_min(nline,y1+2),1):
                        for x in range(int_max(0,x1-1),int_min(ncol,x1+2),1):
                            if zone_rast[y,x]==1:
                                if y!=y1 and x!=x1: Dist = cost_rast[y,x]*diag+dist_ac
                                else: Dist = cost_rast[y,x]*direct+dist_ac                                
                                if Dist <= Max_distance:
                                    if Out_distance[y,x]==101:
                                        Out_distance[y,x] = int(Dist+0.5)
                                        Lien_RF[y,x] = Lien_RF[y1,x1]
                                        Lien_piste[y,x] = Lien_piste[y1,x1]
                                        Dpiste[y,x] = Dpiste[y1,x1]
                                        Dforet[y,x] = Dforet[y1,x1]
                                    else:
                                        if Out_distance[y,x]+0.1*Dpiste[y,x]+Dforet[y,x]>Dist+0.1*Dpiste[y1,x1]+Dforet[y1,x1]:
                                            Out_distance[y,x] = int(Dist+0.5)                                
                                            Lien_RF[y,x] = Lien_RF[y1,x1]
                                            Lien_piste[y,x] = Lien_piste[y1,x1]
                                            Dpiste[y,x] = Dpiste[y1,x1]
                                            Dforet[y,x] = Dforet[y1,x1]                                
        if test==1:count_sans_match = 0
        else:count_sans_match +=1
        dist_ac +=1
        h = int_max(0,h-1)
        b = int_min(nline,b+1)
        l = int_max(0,l-1)
        r = int_min(ncol,r+1)
    for y in range(0,nline,1):
        for x in range(0,ncol,1):
            if Out_distance[y,x]>100:
                Out_distance[y,x]=-9999
                Lien_RF[y,x] = -9999
                Lien_piste[y,x] = -9999
                Dpiste[y,x] = -9999
                Dforet[y,x] = -9999
    return Out_distance,Lien_RF,Lien_piste,Dpiste,Dforet
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef fill_Link(np.ndarray[dtype32_t,ndim=2] Lien_foret_piste, np.ndarray[dtype_t,ndim=2] Lien_piste,np.ndarray[dtype_t,ndim=2] Lien_RF,
                np.ndarray[dtype32_t,ndim=2] Lien_foret_RF, unsigned int nrows,unsigned int ncols):
    cdef unsigned int y=0,x=0
    cdef int pixel,pixel2
    cdef np.ndarray[dtype32_t,ndim=2] Lien_foret_res_pub=np.ones((nrows,ncols),dtype=np.int32)*-9999
    cdef np.ndarray[dtype8_t,ndim=2] Keep=np.zeros((nrows,ncols),dtype=np.int8)
    for y from 0<=y<nrows:
        for x from 0<=x<ncols:
            pixel=Lien_foret_piste[y,x]
            if pixel>0:
                Keep[Lien_piste[pixel,0],Lien_piste[pixel,1]]=1
                Lien_foret_res_pub[y,x]=Lien_piste[pixel,5]
                Lien_foret_RF[y,x]=Lien_piste[pixel,4]
            else:
                pixel2=Lien_foret_RF[y,x]
                if pixel2>0:
                    Keep[Lien_RF[pixel2,0],Lien_RF[pixel2,1]]=1
                    Lien_foret_res_pub[y,x]=Lien_RF[pixel2,3]
    return Lien_foret_res_pub,Lien_foret_RF,Keep
    
