import re, sys
for path in sys.argv[1:]:
    rows=[]
    for l in open(path):
        m=re.match(r'(\S+)\s+(.*?)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)$',l.rstrip())
        if m: rows.append((m[1],m[2],float(m[3])))
    tot=sum(r[2] for r in rows)
    def k(op,nm):
        if 'pre_conv' in nm or nm.startswith('[f16 9x') or nm.startswith('[f16 256x') or 'x[40x' in nm or 'x[20x' in nm or 'x[10x' in nm or '3x3]x[80' in nm or 'q8_0 2560' in nm or ('x[60' in nm and op=='IM2COL'): return 'subsampling front end'
        if op=='MUL_MAT' and nm.startswith('[q8_0'): return 'weight matmuls'
        if 'attn' in nm or op=='SOFT_MAX': return 'attention (scores, softmax, copies)'
        if op=='SIGMOID': return 'GLU gate (sigmoid)'
        if op=='CONT': return 'other copies'
        return 'other small ops'
    cat={}
    for r in rows: cat[k(r[0],r[1])]=cat.get(k(r[0],r[1]),0)+r[2]
    print(path.split('/')[-1], 'listed %.0f ms over 3 runs'%tot)
    for n,v in sorted(cat.items(),key=lambda x:-x[1]): print('   %-40s %5.1f%%'%(n,100*v/tot))
