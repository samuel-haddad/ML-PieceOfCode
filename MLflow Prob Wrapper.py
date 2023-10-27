# The predict method of sklearn's model returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

from mlflow.models.signature import infer_signature

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:,1]


##################################################################################################
wrappedModel = SklearnModelWrapper(model)
# Log the model with a signature that defines the schema of the model's inputs and outputs. 
# When the model is deployed, this signature will be used to validate inputs.
signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
mlflow.pyfunc.log_model("xgboost", python_model=wrappedModel, signature=signature)
##################################################################################################

##################################################################################################
# predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
predictions_train = model.predict_proba(X_train)[:,1]
#################################################################################################

#################################################################################################
# predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
predictions_val = model.predict_proba(X_val)[:,1]
#################################################################################################

#################################################################################################
# predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
predictions_test = model.predict_proba(X_test)[:,1]
#################################################################################################
