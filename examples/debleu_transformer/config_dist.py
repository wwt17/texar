hserver = '192.168.20.47'
h2 = '192.168.20.109'
xiaodan = '192.168.11.89'
xiaodan2 = '10.20.41.2'
daoyuan = '10.20.41.21'
f = '{}:{}'.format
cluster = {
    'ps': [
        f(h2, 40000),
    ],
    'worker': [
        f(h2, 50000),
        f(hserver, 50000),
        f(hserver, 50001),
        f(xiaodan, 50000),
        f(xiaodan, 50001),
        f(daoyuan, 50000),
        f(daoyuan, 50001),
    ],
}
