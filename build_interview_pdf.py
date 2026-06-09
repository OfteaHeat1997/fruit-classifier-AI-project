# -*- coding: utf-8 -*-
"""Generate the interview-prep PDF for Maria Paula (LiveWall AI Developer internship)."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Preformatted,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT

# ----------------------------------------------------------------------------
# Colors / theme
# ----------------------------------------------------------------------------
INK      = colors.HexColor("#1a1a1a")
ACCENT   = colors.HexColor("#0B6E4F")   # green (fruit / livewall vibe)
ACCENT2  = colors.HexColor("#C9F24D")   # lime highlight
LIGHT    = colors.HexColor("#F3F6F2")
CODEBG   = colors.HexColor("#F0F2F4")
GREY     = colors.HexColor("#555555")
BOXQ     = colors.HexColor("#EAF3EF")

styles = getSampleStyleSheet()

def S(name, **kw):
    base = kw.pop("parent", styles["Normal"])
    return ParagraphStyle(name, parent=base, **kw)

body   = S("body", fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK, spaceAfter=6)
bodyJ  = S("bodyJ", parent=body, alignment=TA_LEFT)
h1     = S("h1", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=ACCENT, spaceBefore=4, spaceAfter=4)
h2     = S("h2", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=4)
h3     = S("h3", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=9, spaceAfter=2)
small  = S("small", fontName="Helvetica", fontSize=9, leading=12, textColor=GREY)
bullet = S("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
qstyle = S("qstyle", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=INK, spaceAfter=2)
astyle = S("astyle", fontName="Helvetica", fontSize=10, leading=14, textColor=INK)
codest = S("codest", fontName="Courier", fontSize=8.6, leading=11.5, textColor=INK)
titlel = S("titlel", fontName="Helvetica-Bold", fontSize=27, leading=30, textColor=INK)
subtl  = S("subtl", fontName="Helvetica", fontSize=13, leading=18, textColor=ACCENT)

story = []

def P(text, st=body):
    story.append(Paragraph(text, st))

def B(text):
    story.append(Paragraph(text, bullet, bulletText="•"))

def H1(text):
    story.append(Paragraph(text, h1))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT2, spaceBefore=2, spaceAfter=8))

def H2(text):
    story.append(Paragraph(text, h2))

def H3(text):
    story.append(Paragraph(text, h3))

def SP(h=6):
    story.append(Spacer(1, h))

def CODE(text):
    rows = [[Preformatted(text, codest)]]
    t = Table(rows, colWidths=[165*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), CODEBG),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#D5DADF")),
        ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    story.append(t)
    SP(6)

def QA(q, a):
    inner = [[Paragraph("Q: " + q, qstyle)], [Paragraph("A: " + a, astyle)]]
    t = Table(inner, colWidths=[165*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BOXQ),
        ("LINEBEFORE", (0,0), (0,-1), 3, ACCENT),
        ("LEFTPADDING",(0,0),(-1,-1),9), ("RIGHTPADDING",(0,0),(-1,-1),9),
        ("TOPPADDING",(0,0),(-1,0),7), ("BOTTOMPADDING",(0,-1),(-1,-1),7),
        ("TOPPADDING",(0,1),(-1,1),1),
    ]))
    story.append(KeepTogether(t))
    SP(7)

def TWOCOL(rows, c1="Their requirement", c2="How your project answers it"):
    data = [[Paragraph("<b>"+c1+"</b>", small), Paragraph("<b>"+c2+"</b>", small)]]
    for a, b in rows:
        data.append([Paragraph(a, astyle), Paragraph(b, astyle)])
    t = Table(data, colWidths=[63*mm, 102*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCD5CE")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),7), ("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(t)
    SP(8)

# ============================================================================
# COVER
# ============================================================================
SP(60)
story.append(Paragraph("Interview Preparation Guide", subtl))
SP(6)
story.append(Paragraph("LiveWall &amp; MACH8", titlel))
story.append(Paragraph("AI Developer Internship", titlel))
SP(16)
story.append(HRFlowable(width="40%", thickness=3, color=ACCENT2, hAlign="LEFT"))
SP(16)
P("<b>Candidate:</b> Maria Paula Salazar Agudelo", body)
P("<b>Project used as proof of work:</b> Fruit Ripeness Classifier (deep learning, computer vision)", body)
P("<b>Minor:</b> AI &amp; Society", body)
SP(20)
box = Table([[Paragraph(
    "<b>How to use this guide.</b> Part 1 prepares you to talk about your project in the "
    "interview and connect it to exactly what LiveWall asked for. Part 2 makes sure you truly "
    "understand the foundations, the syntax, and the <i>why</i> behind every choice in your own "
    "code &mdash; so you can answer follow-up questions with confidence. You will not know "
    "everything, and that is fine: LiveWall explicitly wants someone who learns by doing and "
    "is not afraid to experiment.", astyle)]], colWidths=[165*mm])
box.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),LIGHT),
    ("LINEBEFORE",(0,0),(0,-1),3,ACCENT),
    ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
]))
story.append(box)
story.append(PageBreak())

# ============================================================================
# PART 1
# ============================================================================
H1("Part 1 &mdash; The Internship &amp; How to Talk About Your Project")

H2("1.1  What LiveWall (MACH8) is actually looking for")
P("The vacancy is written in Dutch. Here is what it really says, decoded:")
B("<b>MACH8 is LiveWall's AI hub.</b> Their mission: help organizations not just <i>understand</i> AI, but <i>apply</i> it &mdash; building solutions that make impact, from tools and automations to innovative products and experiences.")
B("<b>What you would do:</b> help build AI tools, <b>agents</b> and prototypes; work with <b>LLMs and AI APIs</b>; experiment with new AI tech; support <b>proof-of-concepts</b>; think along on technical solutions; collaborate with engineers.")
B("<b>Who they want:</b> a <i>curious maker</i> who likes to try things out and grow toward an <b>AI Engineer</b> role; a student (HBO/WO), analytical and solution-oriented, <b>not afraid to make mistakes and experiment</b>, with <b>basic programming knowledge</b> and <b>experience with side-projects or own builds.</b>")
B("<b>What they offer:</b> real AI projects, lots of room to learn and experiment, mentorship from experienced engineers, a path to grow into AI Engineer, EUR 500/month, free lunch, 40 hrs/week.")

H2("1.2  The honest gap &mdash; and how to turn it into your strength")
P("Read the vacancy carefully: their day-to-day leans toward <b>generative AI</b> &mdash; LLMs, AI APIs, agents. Your Fruit Classifier is <b>classic computer-vision deep learning</b>. These are different branches of AI. Do not hide this. Handle it like this:")
B("<b>Lead with what matches perfectly:</b> they want \"experience with side-projects or own builds.\" Your project <i>is</i> exactly that &mdash; an end-to-end build you made yourself.")
B("<b>Show the transferable foundations:</b> the core skills carry over &mdash; understanding models, training, evaluation, preprocessing data, calling a model through an <b>API</b>, and shipping something usable. You already wrapped your model in a Flask <b>API</b>; that is conceptually the same shape as working with an AI API.")
B("<b>Show hunger to grow into LLMs:</b> say openly that your project taught you the deep-learning foundations, and you are excited to apply that same learning-by-doing approach to LLMs and agents. That sentence aligns word-for-word with their \"Over jou.\"")
P("<i>This honesty is a feature, not a weakness &mdash; they literally said \"not afraid to make mistakes and experiment.\"</i>", small)

H2("1.3  Your 30-second pitch")
box = Table([[Paragraph(
    "\"I built a Fruit Ripeness Classifier &mdash; a deep-learning app that looks at a photo of a "
    "fruit and tells you whether it is fresh, unripe, or rotten. I took it from a raw dataset all "
    "the way to a working web app: I prepared the data, trained a model using transfer learning "
    "with MobileNetV2, evaluated it, and then wrapped it in a Flask API so anyone could upload a "
    "photo and get an instant prediction. What I enjoyed most was the part beyond accuracy &mdash; "
    "actually turning a trained model into a tool a real person can use. I did it as a personal "
    "side-project to learn by building, and now I want to bring that same hands-on energy to "
    "LLMs and AI agents at MACH8.\"", astyle)]], colWidths=[165*mm])
box.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#1a1a1a")),
    ("TEXTCOLOR",(0,0),(-1,-1),colors.white),
    ("LEFTPADDING",(0,0),(-1,-1),11),("RIGHTPADDING",(0,0),(-1,-1),11),
    ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
]))
story.append(box)
SP(6)
P("Practice saying it out loud until it feels natural &mdash; not memorized word-for-word, but comfortable.", small)

H2("1.4  Mapping your project to what they asked for")
TWOCOL([
    ("Experience with side-projects / own builds",
     "The whole project is a self-initiated build &mdash; dataset to deployed web app."),
    ("Work with AI APIs",
     "You built a REST API with Flask (<font face='Courier'>/api/predict</font>) that receives an image and returns a JSON prediction &mdash; same pattern as consuming/serving AI APIs."),
    ("Build tools &amp; prototypes / proof-of-concepts",
     "Your web app is a working proof-of-concept: upload a photo, get ripeness in seconds."),
    ("Learns by doing, not afraid to experiment",
     "You used data augmentation and transfer learning, iterated, and read the training curves to check for overfitting."),
    ("Analytical &amp; solution-oriented",
     "You framed a real problem (choosing good fruit) and chose a lightweight, mobile-friendly model to solve it."),
    ("Interest in AI, basic programming",
     "Python, TensorFlow/Keras, NumPy, Flask, SQLite, notebooks &mdash; full pipeline in code."),
], c1="What LiveWall asked for", c2="Your evidence (say this)")

H2("1.5  Likely interview questions &amp; strong answers")
QA("Can you walk me through your project?",
   "Use the 30-second pitch, then offer to go deeper: \"It has three stages &mdash; data preparation, model training with transfer learning, and deployment as a Flask API. I can dive into any of them.\" Let them pick &mdash; this shows structure.")
QA("Why did you choose this problem?",
   "\"It is a small everyday problem everyone relates to &mdash; you cannot always tell good fruit from a glance. I liked that it was concrete, visual, and something I could realistically ship end-to-end as a learning project.\"")
QA("What was the hardest part / what did you struggle with?",
   "Be honest and specific: \"Making sure the model generalized instead of just memorizing. I watched the training vs validation curves and used data augmentation and dropout to keep it honest. Also, wiring the trained model into a working API was new to me.\" (Struggling + fixing it is exactly what they want to hear.)")
QA("Your project is computer vision, but we work a lot with LLMs. How do you see that?",
   "\"Right &mdash; my project is classic deep learning, and your focus is generative AI. The branch is different but the foundations transfer: understanding models, prompting/inputs, calling models through APIs, evaluating outputs, and shipping something usable. I built my own API around a model, so I am comfortable with that shape, and I am genuinely excited to learn LLMs and agents hands-on &mdash; which is exactly how I taught myself this project.\"")
QA("What would you improve if you had more time?",
   "\"Three things: finish the TensorFlow Lite conversion so it runs on a phone, collect real user photos to test on messier images, and add proper monitoring of predictions. I already log predictions to a small database as a first step.\"")
QA("How do you learn new technology?",
   "\"By building. For this project I did not start from a course &mdash; I picked a goal, broke it into steps, and learned each piece (transfer learning, augmentation, Flask) as I hit it. That is also why this internship appeals to me: learning by doing with mentorship.\"")

H2("1.6  Smart questions to ask THEM")
B("\"What does a typical proof-of-concept at MACH8 look like &mdash; from idea to demo?\"")
B("\"Which LLMs or frameworks does the team use most right now (e.g. for agents)?\"")
B("\"What would a successful first few months look like for an intern here?\"")
B("\"How much do interns get to experiment versus work on client deliverables?\"")
P("Asking about <i>their</i> tools shows you are already thinking like part of the team.", small)

H2("1.7  Quick prep checklist (the day before)")
B("Re-read the vacancy and this guide once; say the pitch out loud 3x.")
B("Have your GitHub repo open and be ready to screen-share the web app and the training script.")
B("Pick <b>one</b> thing you are proud of and <b>one</b> thing you struggled with &mdash; have both ready.")
B("Skim Part 2 so the <i>why</i> behind each choice is fresh.")
B("Prepare a one-line answer to \"Why LiveWall / MACH8?\" &mdash; tie it to applying AI for real impact and growing toward AI Engineer.")

story.append(PageBreak())

# ============================================================================
# PART 2
# ============================================================================
H1("Part 2 &mdash; Understanding Your Project: Foundations, Syntax &amp; Why")

H2("2.1  The big picture in one paragraph")
P("Your project is an <b>image classifier</b>. You showed a neural network thousands of labeled fruit photos so it could learn the visual patterns of each of 9 categories (fresh / unripe / rotten &times; apple / banana / orange). Instead of training a network from zero, you took <b>MobileNetV2</b> &mdash; a network already trained on millions of images &mdash; and re-used its visual knowledge (<b>transfer learning</b>), adding a small custom \"head\" that learns <i>your</i> 9 classes. After training, the model takes any new photo and outputs 9 probabilities; the highest one is the prediction.")

H2("2.2  Core concepts (in plain language)")
H3("Supervised learning")
P("You learn from <b>labeled examples</b>. Each training image came in a folder named after its class, so the model always knew the \"right answer\" while learning and adjusted itself to reduce its mistakes.")
H3("Classification (vs regression)")
P("The output is a <b>category</b> (which of 9), not a number. That single fact drives later choices: <b>softmax</b> output and <b>categorical crossentropy</b> loss.")
H3("Neural network &amp; CNN")
P("A neural network is layers of simple math units (\"neurons\") that transform the input step by step. For images we use a <b>Convolutional Neural Network (CNN)</b>: convolution layers scan the image for patterns &mdash; edges first, then textures, then shapes &mdash; which is why CNNs are the standard for vision.")
H3("Training, epochs, batches")
P("<b>Training</b> = repeatedly showing examples and nudging the model's internal numbers (<b>weights</b>) to reduce error. One <b>epoch</b> = one full pass over all training images. A <b>batch</b> = the small group of images processed at once (yours: 32) before each update.")

H2("2.3  Why each decision in your code &mdash; the part interviewers probe")

H3("Why deep learning / a CNN at all?")
P("Fruit ripeness is a <b>visual</b> judgment (color, spots, texture). You cannot write simple if-rules for that. A CNN <i>learns</i> those visual features from data, which is exactly what this problem needs.")

H3("Why transfer learning?")
P("Training a strong image model from scratch needs millions of images and lots of GPU time &mdash; you had neither. Transfer learning re-uses a model that already learned generic vision (edges, shapes, colors) and only teaches it the new, small part (your fruit classes). Result: <b>far less data, far less time, much higher accuracy.</b>")

H3("Why MobileNetV2 specifically?")
B("<b>Lightweight &amp; fast</b> &mdash; designed to run on phones, which fits your \"mobile-first\" goal.")
B("<b>Strong accuracy</b> for its size, pre-trained on <b>ImageNet</b> (~1.4M images).")
B("Good trade-off: a heavier model (e.g. ResNet) might be slightly more accurate but too big/slow for a phone.")

H3("Why freeze the base model? (<font face='Courier'>base_model.trainable = False</font>)")
P("Freezing means \"do not change MobileNetV2's learned weights.\" You keep its valuable general vision knowledge and only train your small new head. This prevents wrecking good features with your small dataset, and trains much faster.")

H3("Why resize to 224&times;224?")
P("MobileNetV2 was trained on 224&times;224 images, so it expects that exact input size. Every image &mdash; in training and prediction &mdash; must match, which is why the same size appears in both scripts.")

H3("Why normalize pixels (<font face='Courier'>/ 255.0</font>)?")
P("Pixels come as 0&ndash;255. Neural networks train more stably when inputs are small, similar-scale numbers, so you rescale to 0&ndash;1. Critical: you must apply the <b>same</b> normalization at prediction time, or the model sees \"wrong-looking\" data.")

H3("Why data augmentation (only on training)?")
P("Augmentation makes random variations &mdash; rotation, shift, zoom, horizontal flip &mdash; so the model sees a fruit from many angles/lighting and learns the <i>fruit</i>, not the exact photo. This fights <b>overfitting</b>. You do <b>not</b> augment the test set, because you want to measure performance on clean, real images.")

H3("The custom head, layer by layer")
B("<b>GlobalAveragePooling2D</b> &mdash; squashes MobileNetV2's feature maps into a single flat list of numbers, so a normal Dense layer can use them. (Lighter and less overfit-prone than Flatten.)")
B("<b>Dense(256, relu)</b> &mdash; a fully-connected layer that learns to combine those features into fruit-specific patterns. <b>ReLU</b> is the activation that lets the network model non-linear relationships.")
B("<b>Dropout(0.5)</b> &mdash; during training, randomly switches off half the neurons each step. This stops the model from relying on any single neuron and <b>reduces overfitting</b>.")
B("<b>Dense(9, softmax)</b> &mdash; the final layer: 9 outputs (one per class). <b>Softmax</b> turns them into probabilities that add up to 1, so you can read them as \"how confident\" per class.")

H3("Why these training settings?")
B("<b>Optimizer = Adam</b> &mdash; a smart, adaptive optimizer that adjusts the learning step automatically; a reliable default.")
B("<b>Learning rate = 0.0001 (small)</b> &mdash; because you are fine-tuning on top of a pre-trained model, you take <i>small careful steps</i> so you do not overshoot. Too high = unstable; too low = very slow.")
B("<b>Loss = categorical_crossentropy</b> &mdash; the standard loss for multi-class, one-label classification with softmax. It measures how far the predicted probabilities are from the true label.")
B("<b>Batch size = 32</b> &mdash; a common sweet spot between speed and stable, memory-friendly updates.")
B("<b>Epochs = 20</b> &mdash; enough passes for the curves to flatten out (model stops improving) without over-training.")

H2("2.4  Syntax rules in YOUR code, explained")

H3("The Keras Functional API: <font face='Courier'>x = Layer(...)(x)</font>")
P("This pattern looks strange at first. <font face='Courier'>Dense(256)</font> <i>creates</i> a layer; the second <font face='Courier'>(x)</font> <i>calls</i> it on the previous output <font face='Courier'>x</font>. You are chaining layers, each one taking the previous result &mdash; like a pipeline.")
CODE("x = base_model.output\n"
     "x = GlobalAveragePooling2D()(x)        # feed base output into pooling\n"
     "x = Dense(256, activation='relu')(x)   # then into a dense layer\n"
     "x = Dropout(0.5)(x)                     # then dropout\n"
     "outputs = Dense(9, activation='softmax')(x)\n"
     "model = Model(inputs=base_model.input, outputs=outputs)")

H3("compile / fit / predict &mdash; the three verbs of Keras")
CODE("model.compile(optimizer=Adam(0.0001),       # how it learns\n"
     "              loss='categorical_crossentropy', # what 'wrong' means\n"
     "              metrics=['accuracy'])           # what to report\n\n"
     "history = model.fit(train_generator,          # TRAIN\n"
     "                    epochs=20,\n"
     "                    validation_data=test_generator)\n\n"
     "predictions = model.predict(img_array)        # USE the model")
P("<b>compile</b> = configure how the model learns. <b>fit</b> = actually train it (returns a <font face='Courier'>history</font> of accuracy/loss per epoch, which you plot). <b>predict</b> = run a new image through the trained model.")

H3("Preparing one image for prediction")
CODE("img = Image.open(path).convert('RGB')   # always 3 color channels\n"
     "img = img.resize((224, 224))            # match training size\n"
     "arr = np.array(img).astype('float32') / 255.0   # normalize 0-1\n"
     "arr = np.expand_dims(arr, axis=0)       # (224,224,3) -> (1,224,224,3)")
P("<b>Why <font face='Courier'>expand_dims</font>?</b> The model always expects a <i>batch</i> of images, shaped (batch, height, width, channels). For one image you add a batch dimension of 1.")

H3("Reading the model's answer")
CODE("predictions = model.predict(arr)        # shape (1, 9)\n"
     "idx = np.argmax(predictions[0])         # index of the biggest probability\n"
     "label = class_labels[idx]               # convert index -> fruit name\n"
     "confidence = predictions[0][idx] * 100  # that probability as a %")
P("<b><font face='Courier'>np.argmax</font></b> returns the <i>position</i> of the largest value &mdash; i.e. the winning class. You then map that number to a name using the labels file.")

H3("<font face='Courier'>if __name__ == \"__main__\":</font>")
P("This means \"only run this block if the file is executed directly, not when it is imported by another file.\" It lets <font face='Courier'>predict.py</font> be both a runnable script <i>and</i> an importable module (your web app imports its functions).")

H3("<font face='Courier'>flow_from_directory</font> &mdash; how labels are created")
P("You organized images into one folder per class. <font face='Courier'>flow_from_directory</font> reads those folder names as the labels automatically, resizes images, batches them, and feeds them to training &mdash; no manual labeling code needed.")

H2("2.5  The prediction flow, end to end")
P("When someone uploads a photo, this is the exact journey &mdash; be ready to narrate it:")
B("1. <b>Receive</b> the image (file upload or command-line path).")
B("2. <b>Preprocess</b>: convert to RGB, resize to 224&times;224, normalize to 0&ndash;1, add batch dimension.")
B("3. <b>Infer</b>: <font face='Courier'>model.predict()</font> returns 9 probabilities.")
B("4. <b>Pick the winner</b> with <font face='Courier'>np.argmax</font> and read its confidence.")
B("5. <b>Map</b> the index to a human label via <font face='Courier'>class_labels.json</font>.")
B("6. <b>Return / log</b> the result (JSON from the API; a row in <font face='Courier'>predictions.db</font>).")

H2("2.6  How your web app works (this connects to \"AI APIs\")")
P("Your <font face='Courier'>webapp/app.py</font> uses <b>Flask</b> to expose your model over HTTP. This is exactly the \"work with AI APIs\" idea, so know it well:")
B("<b><font face='Courier'>GET /</font></b> &mdash; serves the HTML page where a user picks a photo.")
B("<b><font face='Courier'>POST /api/predict</font></b> &mdash; receives the image, runs the model, returns JSON: predicted label, confidence, and all 9 probabilities.")
B("<b><font face='Courier'>GET /api/health</font></b> &mdash; a simple check that the server and model loaded correctly.")
P("<b>CORS</b> is enabled so a separate front-end (React, a mobile app) could call this API from another origin. <i>Note for honesty:</i> the web app currently loads a <font face='Courier'>.tflite</font> model that is not generated yet &mdash; mention this as known future work if asked (see 2.10).", small)

H2("2.7  Metrics: what the numbers mean")
B("<b>Accuracy</b> &mdash; the share of images predicted correctly. Easy to explain, but not the whole story.")
B("<b>Loss</b> &mdash; a continuous measure of how wrong the probabilities are; the optimizer minimizes it. Should <i>decrease</i> over epochs.")
B("<b>Validation accuracy/loss</b> &mdash; measured on images the model did <i>not</i> train on. This is the honest signal of real performance.")
B("<b>Overfitting</b> &mdash; when training accuracy is high but validation accuracy lags: the model memorized instead of learning. Your defenses: data augmentation, dropout, and a frozen base. You explicitly watch the gap between the two curves.")

H2("2.8  Quick glossary (skim before the interview)")
gloss = [
    ("Transfer learning", "Re-using a pre-trained model and adapting it to a new task."),
    ("ImageNet", "A huge labeled image dataset used to pre-train MobileNetV2."),
    ("Weights", "The internal numbers the model adjusts while learning."),
    ("Epoch", "One full pass over all training data."),
    ("Batch", "Small group of samples processed before each weight update (yours: 32)."),
    ("ReLU", "Activation function; lets the network learn non-linear patterns."),
    ("Softmax", "Turns final outputs into probabilities that sum to 1."),
    ("Dropout", "Randomly disables neurons in training to reduce overfitting."),
    ("Crossentropy", "Loss function for classification; penalizes confident wrong answers."),
    ("Adam", "Adaptive optimizer; a solid default for training."),
    ("Inference", "Using a trained model to predict on new data."),
    ("Overfitting", "Memorizing training data; poor on new data."),
]
gdata = [[Paragraph("<b>Term</b>", small), Paragraph("<b>Plain meaning</b>", small)]]
for term, mean in gloss:
    gdata.append([Paragraph("<b>"+term+"</b>", astyle), Paragraph(mean, astyle)])
gt = Table(gdata, colWidths=[42*mm, 123*mm], repeatRows=1)
gt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),ACCENT),
    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#CCD5CE")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT]),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
]))
story.append(gt)
SP(8)

H2("2.9  Likely technical questions &amp; crisp answers")
QA("What is transfer learning and why use it here?",
   "Re-using a model pre-trained on millions of images and adapting it to my 9 classes. I used it because I did not have huge data or GPU time, and it gives much higher accuracy faster.")
QA("Why softmax on the last layer?",
   "It converts the 9 raw outputs into probabilities that sum to 1, so I can read confidence per class and pick the highest with argmax.")
QA("What is dropout doing?",
   "During training it randomly turns off neurons so the model cannot lean on any single one. It reduces overfitting and helps it generalize.")
QA("How do you know it is not just memorizing?",
   "I compare training vs validation curves. If validation tracks training closely, it is generalizing; a big gap would mean overfitting. I also used augmentation and dropout as defenses.")
QA("Why normalize the images?",
   "Networks train more stably with small, consistent input values, so I scale pixels from 0-255 to 0-1 &mdash; and I apply the exact same step at prediction time.")
QA("What would break if you fed a 500x500 image straight in?",
   "The model expects 224x224, so I must resize first; otherwise the input shape would not match and it would error or give garbage.")

H2("2.10  Honest limitations (saying these shows maturity)")
B("<b>TensorFlow Lite not finished</b> &mdash; the mobile model is planned; the script exists but I have not generated the <font face='Courier'>.tflite</font> yet, so the web app should be pointed at the <font face='Courier'>.keras</font> model until then.")
B("<b>Tested on dataset images</b> &mdash; real phone photos (messy backgrounds, odd lighting) would be the true test; collecting those is next.")
B("<b>Only 3 fruits, 3 stages</b> &mdash; the scope is deliberately small to ship something working.")
B("<b>Ripeness is subjective</b> &mdash; the boundary between \"fresh\" and \"unripe\" can be fuzzy even for people, so some confusion is expected.")
SP(4)
P("Knowing the limits of your own work is exactly the analytical, honest mindset LiveWall asked for. Good luck, Maria Paula &mdash; you have got this.", small)

# ============================================================================
# Page furniture
# ============================================================================
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(20*mm, 12*mm, "Interview Prep — Fruit Ripeness Classifier")
    canvas.drawRightString(190*mm, 12*mm, "Page %d" % doc.page)
    canvas.setStrokeColor(ACCENT2)
    canvas.setLineWidth(1)
    canvas.line(20*mm, 14*mm, 190*mm, 14*mm)
    canvas.restoreState()

doc = BaseDocTemplate(
    "/home/user/fruit-classifier-AI-project/Interview_Prep_LiveWall.pdf",
    pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=18*mm, bottomMargin=20*mm,
    title="Interview Preparation - LiveWall AI Developer", author="Maria Paula Salazar Agudelo",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(story)
print("PDF built OK")
