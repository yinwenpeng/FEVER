import json

writefile = codecs.open('system_output.json' ,'w', 'utf-8')
lis=[]
for i in range(100):
	dic={}
	dic['haha']='yes'
	dic['val'] = 0.2
	lis.append(dic)
json.dump(lis, writefile)
writefile.close()
