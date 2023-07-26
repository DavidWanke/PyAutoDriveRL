Hier finden Sie den Code für die Bachelorarbeit "Generalisierende Reinforcement-Learning-Agenten für sicheres autonomes Fahren in urbanen Szenarien" von David Wanke.
Das Projekt wurde auf Windows entwickelt und erfordert einige Bibliotheken. Ich empfehle Anaconda zu benutzen und die requirements mittels dieses Befehls in einer neuen Umgebung zu installieren:
conda env create -f environment.yml

Der Code ist aufgrund von fehlender Zeit größtenteils unkommentiert. Falls Sie mit dem Projekt interagieren wollen, nutzen Sie am besten als Einstiegspunkt die main.py. Das Programm benutzt einen Parser zum Starten verschiedener Modi.
Die Parameter sind leider auch aufgrund des Mangels an Zeit undokumentiert. Aber mit einem Blick in die main.py sollte es auch möglich sein die Bedeutung der einzelnen Programmteile und Parameter herzuleiten.
Das Programm hat in der Theorie 6 Modi, die ausgewählt werden können:

training: 
Dies wurde benutzt um die RL-Agenten zu trainieren und bietet verschiedene Parameter zur Anpassung des Trainings. Dieser Modus wird innerhalb der train.py Datei gestartet.


rendering: 
Hiermit können RL-Agenten gestartet werden und in Echtzeit gerendert werden. Da nicht alle Parameter immer benötigt wurden, wurden diese teilweise einfach im Code, wenn benötigt, geändert. Dies kann in der render.py Datei passieren. Dadurch können auch regelbasierte Agenten gerendert werden.


testing: 
Dies wurde hauptsächlich zum Debuggen benutzt um Memory Leaks im Simulator aufzuspüren.


plotting: 
Hiermit wurden alle Plots in der Arbeit erstellt. Damit können automatisch Trainingsexperimente als auch Evaluationen automatisch dargestellt werden. Dieser Modus wird von der plot.py Datei gestartet.


optimize: 
Dies ist ein veralterter Teil des Programms, der früher zum Finden von Seeds für Kollisionen und der Optimierung des Schilds benutzt wurde.


evaluation:
Hiermit können sowohl regelbasierte als auch die trainierten RL-Agenten evaluiert werden. Alle Evaluationen wurde hiermit durchgeführt. Dieser Modus wird von der evaluate.py Datei gestartet.


-------------------------------------------------------
Beispiel für die Durchführung von Experimenten:

1. Trainieren der Agenten
python main.py training EXPERIMENT_BIG_MAP_OBSTACLES --num_workers=24 --seed=1 --total_steps=25000000 --env_min_obstacle_percent 25 --env_max_obstacle_percent 100

python main.py training EXPERIMENT_MINIMAL_MAP --num_workers=24 --seed=1 --total_steps=25000000 --env_use_minimal_map


Mit den beiden Befehlen können zum Beispiel das zweite und das dritte Experiment nachgestellt werden. Der erste Befehl trainiert einen Agenten für 25M Schritte auf den großen, zufällig generierten Straßennetzen mit dem Seed 1 und 24 Threads. Die Threads sollte nicht die Anzahl der Threads des eigenen Prozessors übersteigen. Der zweite Befehl trainiert einen Agenten auf dem minimalen, festen Straßennetz.
Wenn die Befehle ausgeführt werden, werden die Ordner EXPERIMENT_BIG_MAP_OBSTACLES, EXPERIMENT_MINIMAL_MAP in dem Ordner training > experiments erstellt. Hier werden die Modelle, die Log-Dateien und alles weitere gespeichert.

---

2. Begutachtung des Trainingsverlaufes:
Während dem Training, können die Statistiken mittels Tensorboard live begutachtet werden. Hierfür in diesem Ordner oder dem experiments Odner diesen Befehl ausführen:
tensorboard --logdir .

Die Statistiken aller Experimente können dann unter http://localhost:6006/ gesehen werden.

Mit Plotly können diese Ergebnisse auch gerendert werden:
python main.py plotting experiment --paths "EXPERIMENT_BIG_MAP_OBSTACLES,EXPERIMENT_MINIMAL_MAP"

Die Plots landen in saved_images > graphs > training .

---

3. Darstellung der Agenten:
python main.py rendering default EXPERIMENT_BIG_MAP_OBSTACLES

python main.py rendering default EXPERIMENT_MINIMAL_MAP

Hiermit können die beiden trainierten Agenten visuell in Echtzeit dargestellt werden.

Einstellungen für das Rendering können in src\env\view\constants.py passieren. 
Hier sind ein paar Informationen über die Steuerung mittels Tasten während des Renderns:
W und S: Zoomen
1: Debug Screen wechseln
Links-Mausklick: Objekte anklicken und tracken. Dafür muss man eventuell noch 1 drücken um zu dem Tracking Debug Screen zu kommen.
P: Pausieren
T: Neue Episode starten
R: Rendering ausschalten. Die Umgebung läuft aber weiter.

---

4. Evaluation der Agenten:
python main.py evaluation EXPERIMENT_BIG_MAP_OBSTACLES RL_AGENT --num_workers 6 --num_episodes 1000 --env_fps 24 --agent DQN --model_names "check_model_23777280_steps"
python main.py evaluation EXPERIMENT_BIG_MAP_OBSTACLES IDM_AGENT --num_workers 6 --num_episodes 1000 --env_fps 24 --agent IDM --no-env_shield_include_layer_3 --model_names "check_model_23777280_steps"
python main.py evaluation EXPERIMENT_BIG_MAP_OBSTACLES TTC_CREEP_AGENT_12 --num_workers 6 --num_episodes 1000 --env_fps 24 --agent TTC_CREEP --env_agent_ttc_thresh 12.0 --model_names "check_model_23777280_steps"

python main.py evaluation EXPERIMENT_MINIMAL_MAP RL_AGENT --num_workers 6 --num_episodes 1000 --env_fps 24 --agent DQN --model_names "model,check_model_23777280_steps"

Es können nun sowohl regelbasierte Agenten sowie RL-Agenten evaluiert werden. In dem Ordner des dazugehörigen Experiments entsteht dann ein weiterer Ordner evaluation, der die verschiedenen Evaluationen enthält. Mit --model_names können theoretisch mehrere verschiedene durch Komma getrennte Modelle des RL-Agenten geladen werden. Die Modelle sind in dem model Ordner des Experiments zu finden. Das letzte Modell heißt immer "model". 

Nach Ausführung des letzten Befehls wäre dann in dem Ordner experiments > EXPERIMENT_MINIMAL_MAP > evaluation die zwei Ordner "RL_AGENT_m.model_e.1000_fps.24" und "RL_AGENT_m.check_model_23777280_steps_e.1000_fps.24". Dabei handelt es sich um die Evaluation der beiden verschiedenen Modelle des RL-Agenten.

Das Hinzufügen des Namens eines Experiments bei den --model_names erlaubt es auch Agenten aus anderen Experimenten unter den Bedingungen des vorliegenden Experiments zu evaluieren. Dadurch kann zum Beispiel der Agent, der auf der minimalen Karte gefahren ist, auf den großen Straßennetzen des zweiten Experiments evaluiert werden:
python main.py evaluation EXPERIMENT_BIG_MAP_OBSTACLES RL_AGENT --num_workers 6 --num_episodes 1000 --env_fps 24 --agent DQN --model_names "EXPERIMENT_MINIMAL_MAP\model\check_model_23777280_steps"

---

5. Begutachtung der Ergebnisse der Evaluation:
Für das Experiment EXPERIMENT_MINIMAL_MAP:
python main.py plotting evaluation --paths "EXPERIMENT_MINIMAL_MAP\evaluation\RL_AGENT_m.model_e.1000_fps.24,EXPERIMENT_MINIMAL_MAP\evaluation\RL_AGENT_m.check_model_23777280_steps_e.1000_fps.24" --mode SHOW


Mit diesem Befehl können die Ergebnisse der Evaluation der zwei verschiedenen Modelle des Agenten visualisiert werden. Es werden dabei immer die Pfade zu den Evaluationsordnern angegeben. Es können dadurch auch die Evaluationsergebnisse aus verschiedenen Experimenten in einem Plot dargestellt werden.

---

6. Video-Analyse der Kollisionen:
Nach einer Evaluation können alle aufgetretenen Kollisionen dargestellt werden. Hierfür kann dieser Befehl genutzt werden:
python main.py rendering collisions "EXPERIMENT_MINIMAL_MAP\evaluation\RL_AGENT_m.model_e.1000_fps.24"

In dem Ordner "EXPERIMENT_MINIMAL_MAP\evaluation\RL_AGENT_m.model_e.1000_fps.24" wird dann ein Ordner videos angelegt, in dem für jede Kollision ein Video gespeichert wird.
