from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
import pickle
from  sklearn.preprocessing import StandardScaler,OneHotEncoder
application=Flask(__name__)
app=application
scaler=pickle.load(open("pickle/scaler.pkl","rb"))
encoder=pickle.load(open("pickle/encoder.pkl","rb"))
model=pickle.load(open("pickle/rs.pkl","rb"))
@app.route("/")
def hompage():
    return render_template("home.html")

@app.route("/predictdata",methods=["GET","POST"])
def dataprediction():
    if request.method=="POST":
     
       age=float(request.form.get("age"))
       gender=str(request.form.get("gender")) 
       subscription_type=str(request.form.get("subscription_type")) 
       watch_hours=float(request.form.get("watch_hours"))
       last_login_days=float(request.form.get("last_login_days"))
       region=str(request.form.get("region"))
       device=str(request.form.get("device"))
       monthly_fee=float(request.form.get("monthly_fee"))


       num_cols=['age', 'watch_hours', 'last_login_days', 'monthly_fee']
       cat_gorical=['gender', 'subscription_type', 'region', 'device']

       df_num=pd.DataFrame([[age, watch_hours,last_login_days,monthly_fee]],columns=num_cols)
       df_num_scale=scaler.transform(df_num)
       df_categorical=pd.DataFrame([[gender,subscription_type,region,device]],columns=cat_gorical)

       df_cl_en=encoder.transform(df_categorical)
       if hasattr(df_cl_en,"toarray"):
          df_cl_en=df_cl_en.toarray()
       data=np.hstack((df_num_scale,df_cl_en))

       result=model.predict(data)
       return render_template("predict.html",result=result[0])
    else:

        return render_template("predict.html")




if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
