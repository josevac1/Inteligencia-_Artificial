print("Variable")
mensaje = "Hola, mundo!"
n = 15
pi = 3.14159
a= True
b= 4+5j

print (mensaje)
print (n)
print ("hola" , n)

print (type (pi))
#--------------
print("Operadores")
suma= 5+7
sumaFlotante= 5.2+7.41
total= suma + sumaFlotante 
print(suma)
print(sumaFlotante)
print(total)

exponentes= 4**3
print(exponentes)

modulo= 6%2 #0 es par 
print (modulo)
muduloimpar= 5%2
print (muduloimpar)
division=5/2
print(division)
divisionEntera=5//2
print (divisionEntera)
OperadoresLogicos = not True
m=True or False
m2=True and False
m3=True and True and True and True and True 
m4= not (True and False)

print(OperadoresLogicos)
print(m)
print(m2)
print(m3)
print(m4)

print("Tipo de datos")

q=5
print(type(q))
q1=2.3
print(type (q1))
q2= 2.1 +7.8j
print(type(q2))
q4=True
print(type(q4))

cadena ='Hola , mi nombre es Jose y estamos En la Verga y no pondemos mas'
print (cadena)
cadena1='Hola \nMinombre es Jose y esatamos \t Programacion'
print(cadena1)
cadena2=""""Hola,
Mi nombre es Pedro y estamos en :\t Programacion"""
print(cadena2)

un="uno" ; b="dos"
c= un + b
print(c)

un1 =un *6
print (un1)

list=['a','b',[4,10,11],['c',[1,66,['hola']],2,111],'e',7]
print(list[3][1][2])

from typing import Any

nested_data= {
        "a":{
            "b":"1",
            "c":{
                "d":"2",
                "e":"3",
            },
        },
        "f":4,
        "g":{
            "h":5,
            "i":6,
            "j":7,
            "k":8,
        },
}
separator: str="/"
result: dict[str,Any]={}
def flatten (current_value: Any, current_key:str) -> None:
    
    if isinstance(current_value, dict):
        for key, value in current_value.items():#defaul first value
            # Convertimos la clave a string por seguridad
            key_str = str(key)
            #print(key_str)

            value_str = str(value)
            #print(value_str)

            # Construimos la nueva clave acumulando niveles (subniveles)
            new_key = (
                f"{current_key}{separator}{key_str}"
                if current_key
                else key_str
            )
            print("chain: "+new_key)

            # # Llamada recursiva para seguir bajando niveles
            flatten(value, new_key)
        return
    
    # Aquí es donde realmente "aplanamos"
    # Guardamos la clave completa con su valor
    result[current_key] = current_value
    print(current_key," = ",current_value)

# Llamamos a la función
flattened = flatten(nested_data, "")

print("\ndict completo:\n")
print(result)