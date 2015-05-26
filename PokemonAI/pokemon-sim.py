#write the simulator here
class pokemon:
    #this object defines a Pokemon and its attributes.
    #the type of the Pokemon
    #attributes of a Pokemon: stats, ability, moveset, effort values, nature
    '''
    fill in code here!
    '''

class moveset:
    def __init__(self, move1, move2, move3, move4):
        moveset.slot1 = move1
        moveset.slot2 = move2
        moveset.slot3 = move3
        moveset.slot4 = move4

class move:
    def __init__(self, moveFunction, typeOfMove):
        #the function includes what must be done during battle, whether
        #the move is physical/special/other, the power/accuracy of the move
        #the status effect, and the status effect rate. The function is a
        #function of 2 Pokemon.
        move.function = moveFunction
        move.typeOfMove = typeOfMove
