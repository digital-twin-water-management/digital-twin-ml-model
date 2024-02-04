## Digital_Twin_Over_Blockchain_ML_MODEL

### Our first model is XGBoost, which is one of the best boosting algorithms and a good model in itself. We decided this model after comparing with svr (support vector regressor ) and random forest regressor.
### Our second model involves stacking multiple boosting algorithms to improve the accuracy of the model compared to our first model. However, it takes more than 24 hours to train. So we havent got result due to lack of resouces. We consider this model still at experimental stage.

* run boys.py and girls.py only
* Load the dataset and train a model in local laptop without using any cloud library or SageMaker.
* Upload the trained model file to AWS SageMaker and deploy there. Deployment includes placing the model in S3 bucket, creating a SageMaker model object, configuring and creating the endpoints, and a few server-less services (API gateway and Lambda ) to trigger the endpoint from outside world.
* Use a local client ( We use Postman ) to send a sample test data to the deployed model in the cloud and get the prediction back to the client. Restful htttp methods come to our help on this.
