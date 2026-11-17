import random,math
H=open('style.css').read()+open('clay.css').read()
LOGO='<svg width="30" height="30" viewBox="0 0 120 120"><circle cx="52" cy="52" r="40" fill="#fff" stroke="#0F766E" stroke-width="10"/><line x1="82" y1="82" x2="108" y2="108" stroke="#0F766E" stroke-width="12" stroke-linecap="round"/><path d="M30 36h44a6 6 0 0 1 6 6v20a6 6 0 0 1-6 6H50l-12 10v-10h-8a6 6 0 0 1-6-6V42a6 6 0 0 1 6-6z" fill="#E4F1EF" stroke="#0F766E" stroke-width="3"/><circle cx="38" cy="52" r="4.5" fill="#2F8A5B"/><circle cx="52" cy="52" r="4.5" fill="#A3ABA8"/><circle cx="66" cy="52" r="4.5" fill="#C2453D"/></svg>'
def page(name,body,w,extra=""):
    open(f'{name}.html','w').write(f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width={w}"><style>{H}{extra}</style></head><body style="width:{w}px">{body}</body></html>')
def top(active="Analyze"):
    return f'<div class="top"><div class="wm">{LOGO}CommentLens</div><nav><span>{"<b>Analyze</b>" if active=="Analyze" else "Analyze"}</span><span>Saved videos</span><span>Settings</span></nav></div>'
def pill(l,n,on=False,c=None):
    tcls='t-'+l.split()[0].lower()
    d=f'<span class="dot" style="background:{c}"></span>' if c and not on else ''
    return f'<span class="pill {tcls}{" on" if on else ""}">{d}{l} <span class="n">{n}</span></span>'
POS,NEG,NEU,MIX,Q='var(--pos)','var(--neg)','var(--neu)','var(--mix)','var(--ques)'
def moodbar(vals,h=52):
    tot=sum(v[1] for v in vals)
    bar=''.join(f'<div class="seg" style="flex:{v[1]};background:{v[3]};height:{h}px"></div>' for v in vals)
    lab=''.join(f'<div class="seglab" style="flex:{v[1]};min-width:0"><b>{v[0]}</b>{v[1]:,} · {round(v[1]*100/tot)}%</div>' for v in vals).replace(' · ','&nbsp;&nbsp;')
    lab=''.join(f'<div class="seglab" style="flex:{v[1]};min-width:0;white-space:nowrap"><b>{v[0]}</b>{v[1]:,} &nbsp; {round(v[1]*100/tot)}%</div>' for v in vals)
    return f'<div class="mood">{bar}</div><div style="display:flex;gap:4px">{lab}</div>'
def row(a,t,l,s,sc,flags,pr,sel=False):
    fl=''.join(f'<span class="chip {c}">{x}</span>' for x,c in flags)
    return f'<div class="r{" sel" if sel else ""}"><span class="au">{a}</span><span>{t}</span><span class="lk">{l}</span><span class="sent"><span class="dot" style="background:{sc}"></span>{s}</span><span>{fl}</span><span>{pr}</span></div>'
HEAD='<div class="r h"><span>Author</span><span>Comment</span><span style="text-align:right">Likes</span><span>Sentiment</span><span>Labels</span><span>Priority</span></div>'
NEGR=[("@rohit_77","The audio cuts out around 4:12. I had to restart the video twice.","96","Negative",NEG,[("Bug report","b")],"High"),
("@lena_m","Too long. The intro could be half as slow, honestly.","74","Negative",NEG,[("Criticism","")],"Medium"),
("@tomx","Expected more detail on the second half, it felt rushed.","51","Negative",NEG,[("Criticism","")],"Medium"),
("@dani_o","Thumbnail says a $500 build and the video ends at $800.","48","Negative",NEG,[("Criticism","")],"Medium"),
("@s_nair","Why is every shot out of focus after the intro?","37","Negative",NEG,[("Question","q")],"High"),
("@kabir9","Not for beginners. You skipped every setup step.","29","Negative",NEG,[("Suggestion","s")],"Medium"),
("@hugo.p","The link in the description is broken.","19","Negative",NEG,[("Bug report","b")],"High")]
ALLR=[("@maya_k","Best explanation of an editing setup I have seen. Thank you.","412","Positive",POS,[("Praise","s")],"Low"),
("@dev_arjun","Will there be a part 2 covering colour grading?","268","Positive",POS,[("Question","q"),("Suggestion","s")],"High"),
("@sana.reads","Subscribed after this video. Please keep going.","190","Positive",POS,[("Praise","s")],"Low"),
("@priya.s","Which microphone did you use for the voiceover?","131","Neutral",NEU,[("Question","q")],"Medium"),
("@rohit_77",NEGR[0][1],"96","Negative",NEG,[("Bug report","b")],"High")]
SUMMARY="Most viewers liked the editing breakdown and the final result, and many call it the clearest guide they have watched. The main complaints are the slow first five minutes and audio that drops near 4:12. Viewers are asking for a part 2 on colour grading, and for Hindi subtitles."
THEMES=[("Pacing",64),("Audio quality",41),("Part 2 request",38),("Beginner steps",29),("Sponsor length",22),("Subtitles",19),("Thumbnail",14),("Broken links",11)]
def themes():
    m=THEMES[0][1]
    return ''.join(f'<div style="display:grid;grid-template-columns:130px 1fr 36px;gap:14px;align-items:center;margin:11px 0"><span>{t}</span><span style="height:12px;background:#E2EDE9;border-radius:6px;box-shadow:var(--clay-press)"><i class="bar-fill" style="display:block;height:12px;width:{n*100//m}%;background:linear-gradient(90deg,#4FB8AC,#0F766E);border-radius:6px;box-shadow:2px 2px 6px rgba(15,118,110,.25)"></i></span><span class="tab muted" style="text-align:right">{n}</span></div>' for t,n in THEMES)
def timeline(w=1100,h=210):
    random.seed(4);days=14
    def path(base,amp,ph):
        pts=[(i*(w-40)/(days-1)+20,h-30-(base+amp*math.sin(i/2+ph)+random.random()*18+(80 if i<3 and base>100 else 0))*0.6) for i in range(days)]
        return pts
    P=path(120,45,0);N=path(40,18,1.5)
    def d(pts):return 'M'+' L'.join(f'{x:.0f},{y:.0f}' for x,y in pts)
    grid=''.join(f'<line x1="20" x2="{w-20}" y1="{y}" y2="{y}" stroke="#E6EAE7"/>' for y in (30,80,130))
    lab=''.join(f'<text x="{20+i*(w-40)/(days-1)}" y="{h-6}" font-size="11" fill="#9AA5A2" text-anchor="middle">{["Nov 3","","","Nov 6","","","Nov 9","","","Nov 12","","","Nov 15",""][i]}</text>' for i in range(days))
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="display:block">{grid}<path d="{d(P)} L{w-20},{h-30} L20,{h-30}Z" fill="#2F8A5B" opacity=".08"/><path class="draw" pathLength="1" d="{d(P)}" fill="none" stroke="#2F8A5B" stroke-width="2.5"/><path class="draw" pathLength="1" d="{d(N)}" fill="none" stroke="#C2453D" stroke-width="2.5"/><circle cx="{P[2][0]:.0f}" cy="{P[2][1]:.0f}" r="5" fill="#fff" stroke="#2F8A5B" stroke-width="2.5"/>{lab}<text x="{P[2][0]+12:.0f}" y="{max(P[2][1]-14,14):.0f}" font-size="12" font-weight="600" fill="#16211F">Upload day</text></svg>'
MOOD=[("Positive",580,0,POS),("Neutral",140,0,NEU),("Mixed",60,0,MIX),("Negative",220,0,NEG)]
def palrow(active="All"):
    items=[("All","1,000",None),("Positive","580",POS),("Negative","220",NEG),("Neutral","140",NEU),("Questions","134",Q),("Suggestions","96",None),("Bug reports","11",None),("Praise","402",None),("Duplicates","87",None),("Replies","213",None),("Spam","24",None),("Top liked","",None)]
    return '<div class="pills">'+''.join(pill(l,n,l==active,c) for l,n,c in items)+'</div>'
# 01 landing
body=top()+f'''<div class="wrap" style="padding-top:96px">
<div style="max-width:760px"><h1 style="font-size:60px;line-height:1.04">Read what your viewers actually said.</h1>
<p style="font-size:19px;color:var(--muted);margin-top:20px;max-width:560px;line-height:1.55">Paste a video link. CommentLens reads every comment and sorts it, so you can open the complaints, the questions, or the ideas in one click.</p></div>
<div style="margin-top:44px;display:flex;gap:12px;max-width:900px"><div style="flex:1;height:56px;background:var(--surface);box-shadow:var(--clay-press);border:none;border-radius:16px;padding:0 18px;display:flex;align-items:center;color:var(--faint);font-size:16px">https://www.youtube.com/watch?v=</div><span class="btn p lg">Analyze comments</span></div>
<div style="margin-top:20px;display:flex;gap:32px;align-items:center;max-width:900px;color:var(--muted)"><span>Comments to read</span><div style="flex:1;max-width:320px;position:relative;height:4px;background:#DCE3E0;border-radius:2px"><i style="position:absolute;left:0;width:18%;height:4px;background:var(--accent);border-radius:2px"></i><i style="position:absolute;left:18%;top:-7px;width:18px;height:18px;border-radius:50%;background:#fff;border:2px solid var(--accent);margin-left:-9px"></i></div><b class="tab" style="color:var(--ink)">1,000</b><span class="note">About 25 requests, roughly 2 minutes</span></div>
<div style="margin-top:20px"><span class="btn">Try the sample video</span> <span class="note" style="margin-left:12px">No keys needed. Explore every screen with example comments.</span></div>
<div style="margin-top:84px;max-width:1100px"><div class="h2" style="margin-bottom:6px">What you get back</div><p class="note" style="margin-bottom:22px">A sample result, so you know what to expect.</p>
{moodbar(MOOD)}<div style="margin-top:22px">{palrow("Negative")}</div></div>
<div class="foot" style="margin-top:64px">Comments are sent to Google Gemini for classification and saved only on this computer.</div></div>'''
page('01-landing',body,1440)
# 02 analyzing
steps=[("Fetched 1,000 comments","Done","var(--pos)"),("Cleaned and marked 87 duplicates","Done","var(--pos)"),("Reading batch 14 of 25","Working","var(--accent)"),("Writing the summary","Waiting","var(--faint)")]
st=''.join(f'<div style="display:flex;gap:14px;align-items:flex-start;padding:14px 0;border-top:1px solid var(--line)"><span class="{"pulse" if s=="Working" else ""}" style="margin-top:6px;width:10px;height:10px;border-radius:50%;background:{c}"></span><div style="flex:1"><div style="font-weight:{600 if s=="Working" else 500};color:{"var(--ink)" if s!="Waiting" else "var(--faint)"}">{t}</div></div><span class="note">{s}</span></div>' for t,s,c in steps)
body=top()+f'''<div class="wrap" style="padding-top:80px"><div class="note">Analyzing</div><h1 style="font-size:38px;margin:6px 0 40px;max-width:760px;line-height:1.15">I rebuilt my editing setup for under $800</h1>
<div style="display:grid;grid-template-columns:420px 1fr;gap:80px"><div>{st}<div style="margin-top:22px;height:12px;background:#DCE8E4;border-radius:6px;box-shadow:var(--clay-press)"><i class="bar-fill" style="display:block;width:56%;height:12px;background:linear-gradient(90deg,#4FB8AC,#0F766E);border-radius:6px;box-shadow:2px 2px 6px rgba(15,118,110,.3)"></i></div><p class="note" style="margin-top:12px">Results are saved on this computer, so this video will open instantly next time.</p></div>
<div><div class="h2">Read so far</div><p class="note" style="margin:4px 0 24px">560 of 1,000 comments. The bar grows as each batch finishes.</p>
<div class="mood"><div class="seg" style="flex:324;background:var(--pos)"></div><div class="seg" style="flex:78;background:var(--neu)"></div><div class="seg" style="flex:34;background:var(--mix)"></div><div class="seg" style="flex:124;background:var(--neg)"></div><div class="seg" style="flex:440;background:repeating-linear-gradient(135deg,#E8EDEA 0 8px,#F2F5F3 8px 16px)"></div></div>
<div style="display:flex;gap:40px;margin-top:20px"><div><b class="tab" style="font-size:26px;font-family:'Inter Display'">324</b><div class="note">positive</div></div><div><b class="tab" style="font-size:26px;font-family:'Inter Display'">124</b><div class="note">negative</div></div><div><b class="tab" style="font-size:26px;font-family:'Inter Display'">75</b><div class="note">questions</div></div><div><b class="tab" style="font-size:26px;font-family:'Inter Display'">49</b><div class="note">suggestions</div></div></div></div></div></div>'''
page('02-analyzing',body,1440)
# 03 dashboard
body=top()+f'''<div class="wrap" style="padding-top:44px">
<div style="display:flex;justify-content:space-between;align-items:flex-end;gap:40px"><div><h1 style="font-size:34px;line-height:1.15;max-width:760px">I rebuilt my editing setup for under $800</h1><div class="muted" style="margin-top:10px;display:flex;gap:28px"><span>Studio Notes</span><span class="tab">184,203 views</span><span class="tab">1,000 comments analyzed</span><span>2 minutes ago</span></div></div><div style="display:flex;gap:10px"><span class="btn">Re-analyze</span><span class="btn p">Download report</span></div></div>
<div style="display:grid;grid-template-columns:200px 1fr;gap:48px;margin-top:44px;align-items:end"><div><div class="tab" style="font-family:'Inter Display';font-size:84px;font-weight:650;letter-spacing:-.04em;line-height:.9">64</div><div style="margin-top:8px;font-weight:600">Mostly positive</div><div class="note">Mood score out of 100</div></div><div>{moodbar(MOOD,60)}</div></div>
<div style="margin-top:36px">{palrow("All")}<div style="display:flex;justify-content:space-between;margin-top:14px"><span class="note">A comment can belong to more than one group, so counts can add up to more than the total.</span><span class="toggle"><i></i>Show toxic comments</span></div></div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:72px;margin-top:56px"><div><div class="h2">What viewers are saying</div><p style="margin-top:12px;font-size:16px;line-height:1.7;max-width:540px">{SUMMARY}</p></div><div><div class="h2">What people keep mentioning</div><div style="margin-top:10px">{themes()}</div></div></div>
<div style="margin-top:56px"><div style="display:flex;justify-content:space-between;align-items:baseline"><div class="h2">Mood by day</div><div class="note" style="display:flex;gap:18px"><span><span class="dot" style="background:var(--pos)"></span> Positive</span><span><span class="dot" style="background:var(--neg)"></span> Negative</span></div></div><div style="margin-top:12px;background:var(--surface);border:none;border-radius:26px;box-shadow:var(--clay);padding:18px 14px 6px">{timeline()}</div><p class="note" style="margin-top:10px">Positive comments peaked on upload day. Negative comments stayed low until the sponsor segment was mentioned.</p></div>
<div style="margin-top:56px;display:flex;justify-content:space-between;align-items:center"><div><div class="h2">Comments</div><div class="note" style="margin-top:2px">Showing 1,000 comments, sorted by likes</div></div><div class="search">Search comments</div></div>
<div class="tbl" style="margin-top:16px">{HEAD}{''.join(row(*r) for r in ALLR)}</div>
<div class="foot">Design reference with example data.</div></div>'''
page('03-dashboard',body,1440)
# 04 explore + drawer
tbl=HEAD+''.join(row(*r,sel=(i==0)) for i,r in enumerate(NEGR))
drawer='''<div style="position:absolute;right:0;top:0;bottom:0;width:470px;background:var(--surface);border-radius:26px;box-shadow:var(--clay);padding:32px 34px;margin:16px 16px 16px 0">
<div style="display:flex;justify-content:space-between;align-items:center"><div class="h2">Comment</div><span class="note">Close</span></div>
<div style="margin-top:22px;display:flex;justify-content:space-between"><b>@rohit_77</b><span class="note tab">Nov 4, 2026 &nbsp; 96 likes &nbsp; 3 replies</span></div>
<p style="margin-top:14px;font-size:18px;line-height:1.55;letter-spacing:-.005em">The audio cuts out around 4:12. I had to restart the video twice.</p>
<div style="margin-top:20px;display:flex;gap:8px;flex-wrap:wrap"><span class="chip b">Bug report</span><span class="chip h">High priority</span><span class="chip">Audio quality</span><span class="chip">English</span></div>
<div style="margin-top:26px;padding-top:20px;border-top:1px solid var(--line)"><div class="note">Sentiment</div><div style="display:flex;align-items:center;gap:14px;margin-top:8px"><span class="tab" style="font-family:'Inter Display';font-size:28px;font-weight:650">-0.72</span><span style="flex:1;height:8px;border-radius:4px;background:linear-gradient(90deg,#C2453D,#E1E6E3 50%,#2F8A5B);position:relative"><i style="position:absolute;left:14%;top:-4px;width:4px;height:16px;background:var(--ink);border-radius:2px"></i></span></div></div>
<div style="margin-top:24px"><div class="note">In short</div><p style="margin-top:4px">Viewer reports audio dropping at 4:12.</p></div>
<div style="margin-top:24px"><div class="note">Suggested reply</div><div style="margin-top:8px;padding:16px;border:none;border-radius:18px;background:var(--mint);box-shadow:var(--clay-press);line-height:1.6">Thanks for flagging this. We found the drop at 4:12 and will upload a fixed version this week.</div><div style="margin-top:12px;display:flex;gap:10px"><span class="btn p">Copy reply</span><span class="btn">Mark as handled</span></div></div></div>'''
body=top()+f'''<div style="position:relative;min-height:940px"><div class="wrap" style="padding-top:40px;max-width:none;margin:0;padding-left:48px;padding-right:520px">
<div style="margin-bottom:26px;display:flex;gap:28px;border-bottom:1px solid var(--line)"><span style="padding:10px 0;border-bottom:2px solid var(--accent);font-weight:600">Explore comments</span><span class="muted" style="padding:10px 0">Ask the comments</span><span class="muted" style="padding:10px 0">Top quotes</span><span class="muted" style="padding:10px 0">Compare videos</span></div>
{palrow("Negative")}
<div style="display:flex;justify-content:space-between;align-items:center;margin-top:28px"><div><div class="h2">220 negative comments</div><div class="note">Sorted by likes. Select a row to read it in full.</div></div><div style="display:flex;gap:10px;align-items:center"><div class="search" style="width:240px">Search comments</div><span class="btn">Download this group</span></div></div>
<div class="tbl" style="margin-top:16px">{tbl}</div></div>
<div style="position:absolute;inset:0 470px 0 0;background:rgba(245,247,246,.55)"></div>{drawer}</div>'''
page('04-explore-drawer',body,1440,'.tbl .r{grid-template-columns:110px 1fr 50px 100px 110px 70px;gap:12px}')
# 05 ask
cites=''.join(row(a,t,l,s,sc,f,p) for a,t,l,s,sc,f,p in [("@dev_arjun","Will there be a part 2 covering colour grading?","268","Positive",POS,[("Question","q"),("Suggestion","s")],"High"),("@kabir9","Please add Hindi subtitles. I share this with my team.","88","Positive",POS,[("Suggestion","s")],"Medium"),("@mei_w","Could you show the full cable layout in a part 2?","61","Neutral",NEU,[("Question","q")],"Medium")])
cc=lambda s:f'<span style="display:inline-block;background:var(--accent-soft);color:#0B5F58;font-weight:600;font-size:12px;padding:1px 8px;border-radius:999px;margin:0 2px;font-variant-numeric:tabular-nums">{s}</span>'
body=top()+f'''<div class="wrap" style="padding-top:40px"><div style="margin-bottom:34px;display:flex;gap:28px;border-bottom:1px solid var(--line)"><span class="muted" style="padding:10px 0">Explore comments</span><span style="padding:10px 0;border-bottom:2px solid var(--accent);font-weight:600">Ask the comments</span><span class="muted" style="padding:10px 0">Top quotes</span><span class="muted" style="padding:10px 0">Compare videos</span></div>
<div style="max-width:780px"><div class="note">You asked</div><h1 style="font-size:34px;line-height:1.2;margin:6px 0 34px">What do viewers want next?</h1>
<div style="font-size:17px;line-height:1.75"><p>The most requested follow-up is a part 2 on colour grading, mentioned in 38 comments {cc("c12")}{cc("c88")}{cc("c140")}. The second most common ask is subtitles, especially in Hindi {cc("c203")}.</p><p style="margin-top:16px">A smaller group wants the full cable layout and desk dimensions shown on screen {cc("c317")}. Nobody asked for a longer video; several asked for a shorter intro {cc("c45")}.</p></div>
<div class="note" style="margin-top:18px">Based on 250 comments chosen for this question. Answers use comment text only.</div></div>
<div style="margin-top:40px"><div class="h2">Comments cited</div><div class="tbl" style="margin-top:14px">{HEAD}{cites}</div></div>
<div style="margin-top:40px;max-width:780px"><div class="note" style="margin-bottom:10px">Try asking</div><div class="pills"><span class="pill">What did people dislike most?</span><span class="pill">Are there bugs mentioned?</span><span class="pill">Which questions should I answer?</span></div>
<div style="display:flex;gap:10px;margin-top:16px"><div style="flex:1;height:52px;border:none;background:var(--surface);box-shadow:var(--clay-press);border-radius:16px;padding:0 18px;display:flex;align-items:center;color:var(--faint);font-size:15px">Ask a question about these comments</div><span class="btn p" style="height:52px">Ask</span></div></div><div class="foot">Design reference with example data.</div></div>'''
page('05-ask-the-comments',body,1440)
# 06 mobile
body=f'''<div class="top" style="padding:0 20px;height:56px"><div class="wm">{LOGO}CommentLens</div><span class="muted">Menu</span></div><div style="padding:24px 20px">
<h1 style="font-size:26px;line-height:1.15">I rebuilt my editing setup for under $800</h1><div class="note" style="margin-top:8px">Studio Notes, 1,000 comments analyzed</div>
<div style="display:flex;gap:16px;align-items:flex-end;margin-top:26px"><div class="tab" style="font-family:'Inter Display';font-size:68px;font-weight:650;letter-spacing:-.04em;line-height:.9">64</div><div><b>Mostly positive</b><div class="note">Mood score out of 100</div></div></div>
<div class="mood" style="margin-top:20px"><div class="seg" style="flex:580;background:var(--pos);height:40px"></div><div class="seg" style="flex:140;background:var(--neu);height:40px"></div><div class="seg" style="flex:60;background:var(--mix);height:40px"></div><div class="seg" style="flex:220;background:var(--neg);height:40px"></div></div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 16px;margin-top:14px"><div class="seglab" style="margin:0"><span class="dot" style="background:var(--pos)"></span> <b style="display:inline">Positive</b> 580</div><div class="seglab" style="margin:0"><span class="dot" style="background:var(--neg)"></span> <b style="display:inline">Negative</b> 220</div><div class="seglab" style="margin:0"><span class="dot" style="background:var(--neu)"></span> <b style="display:inline">Neutral</b> 140</div><div class="seglab" style="margin:0"><span class="dot" style="background:var(--mix)"></span> <b style="display:inline">Mixed</b> 60</div></div>
<div style="margin:26px -20px 0;padding:0 20px;overflow:hidden"><div style="display:flex;gap:8px;white-space:nowrap">{pill("All","1,000",True)}{pill("Positive","580",False,POS)}{pill("Negative","220",False,NEG)}{pill("Questions","134",False,Q)}</div></div>
<p style="margin-top:28px;font-size:15px;line-height:1.65">{SUMMARY}</p>
<div class="h2" style="margin-top:30px">Comments</div>
<div style="margin-top:12px">{''.join(f'<div style="padding:16px 0;border-top:1px solid var(--line)"><div style="display:flex;justify-content:space-between"><b>{a}</b><span class="note tab">{l} likes</span></div><p style="margin:6px 0 10px;font-size:15px">{t}</p><div style="display:flex;align-items:center;gap:8px"><span class="dot" style="background:{sc}"></span><span class="note">{s}</span> {"".join(f"<span class=chip {c}>{x}</span>" for x,c in f)}</div></div>' for a,t,l,s,sc,f,p in ALLR[:4])}</div>
<div style="margin:24px -20px 0;padding:14px 20px;background:var(--paper);border-top:1px solid var(--line);display:flex;gap:10px"><span class="btn" style="flex:1;justify-content:center">Ask the comments</span><span class="btn p" style="flex:1;justify-content:center">Download report</span></div></div>'''
page('06-mobile',body,390)
# 07 design system
sw=lambda n,h,u:f'<div><div style="height:72px;border-radius:10px;background:{h};border:1px solid rgba(0,0,0,.06)"></div><b style="display:block;margin-top:8px">{n}</b><span class="note tab">{h}</span><div class="note">{u}</div></div>'
body=f'''<div class="wrap" style="padding-top:56px;max-width:1280px"><div style="display:flex;align-items:center;gap:14px">{LOGO}<h1 style="font-size:34px">CommentLens design system</h1></div><p class="muted" style="margin-top:8px;max-width:640px">Light, calm, and quiet. One accent color for things you can act on. Sentiment colors only ever mean sentiment.</p>
<div class="h2" style="margin-top:44px">Color</div><div style="display:grid;grid-template-columns:repeat(6,1fr);gap:18px;margin-top:16px">{sw("Paper","#F5F7F6","Page background")}{sw("Surface","#FFFFFF","Table, drawer, inputs")}{sw("Ink","#16211F","Text")}{sw("Muted","#667370","Notes, labels")}{sw("Line","#E1E6E3","1px borders")}{sw("Accent","#0F766E","Active, primary actions")}</div>
<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:18px;margin-top:18px">{sw("Positive","#2F8A5B","Sentiment only")}{sw("Negative","#C2453D","Sentiment only")}{sw("Neutral","#A3ABA8","Sentiment only")}{sw("Mixed","#D4A017","Sentiment only")}{sw("Question","#3B6FD4","Question label")}{sw("Accent tint","#E4F1EF","Selected row, hover")}</div>
<div style="display:grid;grid-template-columns:1.1fr 1fr;gap:64px;margin-top:48px"><div><div class="h2">Type</div><div style="margin-top:16px;border-top:1px solid var(--line)">
{''.join(f'<div style="display:grid;grid-template-columns:150px 1fr;align-items:baseline;padding:14px 0;border-bottom:1px solid var(--line)"><span class="note">{n}</span><span style="{s}">{t}</span></div>' for n,s,t in [("Display 60/62","font:650 44px 'Inter Display';letter-spacing:-.025em","Read what viewers said"),("Title 34","font:650 34px 'Inter Display';letter-spacing:-.025em","Video title"),("Number 84","font:650 56px 'Inter Display';letter-spacing:-.04em;font-variant-numeric:tabular-nums","64"),("Heading 20","font:650 20px 'Inter Display'","What viewers are saying"),("Body 14/15","font:400 15px Inter","Comment text, tables, notes"),("Note 13","font:400 13px Inter;color:#667370","Counts, helper text")])}
</div><p class="note" style="margin-top:12px">Inter for everything. Inter Display for titles and big numbers. Sentence case, no all caps, tabular numbers for every count.</p></div>
<div><div class="h2">Components</div><div style="margin-top:16px;display:grid;gap:22px">
<div><div class="note" style="margin-bottom:8px">Palette pills: default, with sentiment dot, active</div><div class="pills">{pill("Questions","134")}{pill("Negative","220",False,NEG)}{pill("Negative","220",True)}</div></div>
<div><div class="note" style="margin-bottom:8px">Buttons: primary, secondary</div><div style="display:flex;gap:10px"><span class="btn p">Analyze comments</span><span class="btn">Re-analyze</span></div></div>
<div><div class="note" style="margin-bottom:8px">Labels on a comment</div><span class="chip b">Bug report</span><span class="chip q">Question</span><span class="chip s">Suggestion</span><span class="chip h">High priority</span><span class="chip">Audio quality</span></div>
<div><div class="note" style="margin-bottom:8px">Mood bar: segments sized by count, 52 to 60px tall, 4px gaps, 10px radius</div><div class="mood"><div class="seg" style="flex:580;background:var(--pos);height:36px"></div><div class="seg" style="flex:140;background:var(--neu);height:36px"></div><div class="seg" style="flex:60;background:var(--mix);height:36px"></div><div class="seg" style="flex:220;background:var(--neg);height:36px"></div></div></div>
<div><div class="note" style="margin-bottom:8px">Table row, selected row</div><div class="tbl" style="font-size:13px"><div class="r" style="grid-template-columns:90px 1fr 60px;min-height:46px;border-top:0"><b>@tomx</b><span>Expected more detail on the second half.</span><span class="lk">51</span></div><div class="r sel" style="grid-template-columns:90px 1fr 60px;min-height:46px"><b>@rohit_77</b><span>The audio cuts out around 4:12.</span><span class="lk">96</span></div></div></div></div></div></div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:48px;margin-top:48px;padding-bottom:56px"><div><div class="h2">Spacing and shape</div><p style="margin-top:10px;color:var(--muted)">Base unit 8px. Sections 56px apart. Content width 1180px, left aligned. Pills fully round, inputs and buttons 8px, table and drawer 12px. 1px borders, no shadows, no gradients.</p></div><div><div class="h2">Motion</div><p style="margin-top:10px;color:var(--muted)">One moment only: the mood bar fills while batches finish. Everything else answers a click: pill fill, drawer open, copy confirmed.</p></div><div><div class="h2">Avoid</div><p style="margin-top:10px;color:var(--muted)">Emoji, icons used as decoration, all-caps labels, purple or glow effects, identical cards everywhere, dark mode in v1, marketing words.</p></div></div></div>'''
page('07-design-system',body,1360)
