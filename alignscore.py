from AlignScore.src.alignscore import AlignScore

scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt', evaluation_mode='nli_sp')
score = scorer.score(contexts=['hello world.'], claims=['hello world.'])

print("Score: ",score[0])