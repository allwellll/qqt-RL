"""Print strong eval (seats 0,1 vs G_frozen) for league v2 arms and references: python -m scripts.cc_league2_strong_table [filters]"""
import glob, json, sys
P = 'runs/cc_train_20261105/eval_strong_v1/points/'
def main():
    names = ['G_frozen', 'S05_sil_u018862'] + sorted(p.split('/')[-1][:-len('__seats01.json')] for p in glob.glob(P + 'L2_*__seats01.json'))
    names = [n for n in names if not sys.argv[1:] or n in ('G_frozen', 'S05_sil_u018862') or any(f in n for f in sys.argv[1:])]
    print('model'.ljust(18), 'key_vs_G_frozen'.ljust(20), 'greedy W-L-D score self caus deliv | categ W-L-D score self caus deliv')
    for n in names:
        d = json.load(open(P + n + '__seats01.json')); g = d['G_frozen']
        f = lambda m: f"{m['wins']}-{m['losses']}-{m['draws']} {m['score']:.3f} {m['self_deaths']:3d} {m['causal_kills']:3d} {m['deliveries']:3d}"
        print(n.ljust(18), str(d['selection_key_vs_G_frozen']).ljust(20), f(g['greedy']), '|', f(g['categorical']))


if __name__ == '__main__':
    main()
