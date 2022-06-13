from flask import Flask, jsonify,request,json
import nao_rl_framework as nrlf,nao_superviser as ns,time, pickle

robot = ns.Nao()
app = Flask(__name__)

@app.route("/")
def debug():
    return "it's working champ!"

@app.route("/explore",methods=['POST'])
def explore():
    if (request.headers.get('Content-Type') == 'application/json'):
        jpost = request.json
        jobj = json.loads(request.json)
        print(jobj['action'])
        # collect init_cord, goal_cord
        init_cord = jobj['NaoPosition']
        goal_cord = jobj['GoalPosition']
        # for num of searchs
        # set simulation to init state
        # generate randome action
        # run simulation once
        robot.__stepSimulaiton__()
        # collect next_cord
        robot.__storeCurrentState__()
        next_state = robot.__getLastState__()['NaoPosition']
        # call partial reward
        nrlf.partial_reward()
        # compare with best move
        # return partial value of best move alongside best move 
        return jpost
    else:
        return 'input must be initial state of the robot in json format'

if __name__ == "__main__":
    
    app.run(host='127.0.0.1', port=123)