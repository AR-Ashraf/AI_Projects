import pyttsx3
import PyPDF2

book = open('AIbe.pdf', 'rb')

speaker = pyttsx3.init("sapi5")
pdfReader = PyPDF2.PdfFileReader(book)
pages = pdfReader.numPages
voices = speaker.getProperty("voices")
speaker.setProperty("voice", voices[1].id)

for num in range(73, pages):
    page = pdfReader.getPage(num)
    text = page.extractText()
    speaker.say(text)
    speaker.runAndWait()