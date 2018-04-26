from __future__ import unicode_literals

from collections import defaultdict

from sshtunnel import SSHTunnelForwarder
import os
import pymysql

    
with SSHTunnelForwarder(('scrubber.yiad.am'),
         ssh_password="v!18pw",
         ssh_username="vivien",
         remote_bind_address=('127.0.0.1', 3306)) as server:

    connection = pymysql.connect(host='127.0.0.1',
                           port=server.local_bind_port,
                           user='root',
                           passwd='',
                           db='scrubber')
    cursor=connection.cursor()
    sql= "SELECT id, annotatedText from scrubber_sentence WHERE created_at <> updated_at;"


    cursor.execute(sql)
    data=cursor.fetchall()
    for id, text in data:
        print (text)
