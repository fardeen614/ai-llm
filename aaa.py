import cv2
from deepface import DeepFace
from langchain_groq import ChatGroq

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#C:\Users\farde\AppData\Local\Programs\Python\Python310\Scripts\c:/Users/farde/aaa.py     "Run using your python installed path and using the name of the python file

import os


# Initialize LLM
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm


# Create Vector DB
def create_vector_db():
    loader = DirectoryLoader("c:/Users/farde/data", glob='*.pdf', loader_cls=PyPDFLoader)   #create a folder data under same area where your python file is there and under data put the mental health document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    print("ChromaDB created and data saved")
    return vector_db


# Setup QA Chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """ You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    Emotion: {emotion}
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['emotion', 'context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# Real-time Emotion Detection and Chatbot
def detect_emotion_and_chat():
    print("Initializing Chatbot...")
    llm = initialize_llm()
    db_path = "c:/Users/farde/chroma_db"

    if not os.path.exists(db_path):
        vector_db = create_vector_db()
    else:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    qa_chain = setup_qa_chain(vector_db, llm)

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    print("Webcam started. Press 'q' to quit.")
    detected_emotion = "neutral"  # Default emotion

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access webcam. Exiting...")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                detected_emotion = result[0]['dominant_emotion']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception:
                detected_emotion = "neutral"

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Chatbot interaction
        print("\nType your query below:")
        query = input("Human: ")
        if query.lower() == "exit":
            print("Chatbot: Take care of yourself. Goodbye!")
            break

        try:
            response = qa_chain.run({"emotion": detected_emotion, "question": query})
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"Chatbot error: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_emotion_and_chat()
