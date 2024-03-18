import numpy as np

a1 = [1,2,3,4,5,6,7,8,9,10]

#a2 = np.append(a1 , [1,2,3,4,5,6,7,8,9,20])
#a2 = np.stack((a1 , a1))
#a2 = np.vstack((a1 , a1))
a2 = np.hstack((a1,a1))

#print(np.concatenate((a1,a2)))
print(np.hstack((a1,a2)))
print()
a2 = a2.reshape(4,5)
print(a2)

media_pe_linii = np.mean(a2 , axis = 1)
media_pe_col = np.mean(a2 , axis = 0)
print()
print('Mediile pe linii : ' , media_pe_linii)
print('Mediile pe coloane : ' , media_pe_col)
print()
prima_linie = a2[0]

print('Prima linie : ' , prima_linie)
print()

ultimile_doua_linii = a2[2:]

print('Utilimele 2 linii : ' , ultimile_doua_linii)
print()

col_2_3_4 = a2[: , 2:]
print('Coloanele 2 ,3, 4 : ' , col_2_3_4)
print()

col_2_4 = a2[: , [1,3]]
print('Coloanele 2,4 : ' , col_2_4)
print()

elem_pe_liniile_2_3 = a2[[1,2] , :]
print()
elem_pe_coloanele_1_4 = a2[: , 0:]

print('Elementele pe liniile 2 , 3 : ' , elem_pe_liniile_2_3)
print()
print('Elementele pe coloanele 1, 4 : ', elem_pe_coloanele_1_4)
print()

inv_coloane = elem_pe_coloanele_1_4[::-1]
print('Coloanele afisate invers : ' , inv_coloane)
print()

elem_pe_coloanele_1si4 = a2[: , [0,3]]
print('Elementele pe coloanele 1 si 4 : ' , elem_pe_coloanele_1si4)

inv_col2 = elem_pe_coloanele_1si4[::-1]
print('Elemente pe coloanele 1 si 4 , inv : ', inv_col2)
print()

afis_1 = a2[[2,3] , 1:3]
afis_2 = a2[[2,3] , [1,2]]

print('Afisare var 1 : ' , afis_1)
print()
print('Afisare var 2 : ' , afis_2)
print()

filtru = np.eye(a2.shape[0] , a2.shape[1] , dtype=bool)

diagonala_var1 = a2[filtru]

print('Diagonala varianta 1 : ' , diagonala_var1)
print()

diagonala_var2 = np.diag(a2)
print('Diagonala varianta 2 : ', diagonala_var2)
print()

indicii_elementelor = np.where((a2 >=3) & (a2 < 5))
print('Indicii cu where : ' , indicii_elementelor)
print()

tablou_sortat_pe_linii = np.sort(a2 , axis = 1)
tablou_sortat_pe_coloane = np.sort(a2, axis = 0)


print('Sortare pe linii : ' , tablou_sortat_pe_linii)
print()
print('Sortare pe coloane : ' , tablou_sortat_pe_coloane)
print()

redimensionare_1D = a2.reshape(1 , -1)
print('Tablou redimensionat 1D : ' , redimensionare_1D)
print()

tablou_24elem = np.linspace(0 , 23 , 24)
print('Tablou de 24 elemente : ' , tablou_24elem)
print()

tablou_reshape = tablou_24elem.reshape(2,3,4)
print('Tablou 2x3x4 : ' , tablou_reshape)

#np.transpose()