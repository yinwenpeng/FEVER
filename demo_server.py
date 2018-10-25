from demo_test_function import evaluate_lenet5
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse
import socketserver as SocketServer
import pickle
import codecs
root = '/home1/w/wenpeng/dataset/FEVER/'

title2sentlist = ''
title2wordlist = ''
word2id = {}

class S(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        try:
            params = urlparse.parse_qs(urlparse.urlparse(self.path).query)

            # Change parameter names here
            first_name = params["premise"][0];
            # last_name = params["hypothesis"][0];

            # Change your function here
            self.wfile.write(str.encode(evaluate_lenet5(first_name,title2sentlist,title2wordlist,word2id)))
        except:
            self.wfile.write(str.encode(open('demo_index.html', 'r').read()))

class wenpeng(object):
    def __init__(self):
        global title2sentlist
        global title2wordlist
        with open(root+'title2sentlist.p', 'rb') as fp:
            print('open title2sentlist.p...')
            title2sentlist = pickle.load(fp)
        with open(root+'title2wordlist.p', 'rb') as fp:
            print('open title2wordlist.p...')
            title2wordlist = pickle.load(fp)

        global word2id
        read_word2id = codecs.open(root+'word2id.txt', 'r', 'utf-8')
        for line in read_word2id:
            parts = line.strip().split()
            word2id[parts[0]] = int(parts[1])
        print('word2id load over, size:', len(word2id))
        read_word2id.close()
    def run(self, server_class=HTTPServer, handler_class=S, port=4004):
        server_address = ('', port)
        httpd = server_class(server_address, handler_class)
        print('Starting httpd...')
        httpd.serve_forever()

newclass = wenpeng()
newclass.run(port=4007)
