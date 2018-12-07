# PyTorch or TensorFlow for NN
# SkLearn for decision trees
#  Meeting Monday at 1pm room 228 (Whatsapp)
#
# Adjust the rules with the

class Learn:

    def CreateFeaturesVectorTrump(self):
        """
        Needed features:
        - hand (32 binary features)
        - number of passes [0,1,2,3]
        - game score for my team
        - game score for opposite team
        - round number
        """
        pass

    def CreateFeaturesVectorPlay(self):
        """
        Needed features:
        - hand (same as above)
        - which card were played and by who (32 numbers from 0 to 4: 0 not played, 1,2,3,4 number of player)
        - round scores (2 numbers)
        - trump (which suit [1,2,3,4], who chose it [1,2,3,4])
        (- round number and game scores) optional
        """
        pass
    
