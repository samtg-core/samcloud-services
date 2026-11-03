import ast,asyncio,json,sys,types,unittest
from pathlib import Path
SOURCE=(Path(__file__).parent/'server.py').read_text()
tree=ast.parse(SOURCE)
node=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='stream' and any(isinstance(v,ast.Name) and v.id=='saw_tool_calls' for v in ast.walk(n)))
fix=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fix_tool_calls')
class Cases(unittest.IsolatedAsyncioTestCase):
 async def frames(self,chunks):
  async def chat_stream(*a,**kw):
   for chunk in chunks:yield chunk
  ns=dict(json=json,asyncio=asyncio,mgr=types.SimpleNamespace(ollama=types.SimpleNamespace(chat_stream=chat_stream)),mm=types.SimpleNamespace(name='fixture'),ollama_messages=[],n_prompt=0,queue_owned=True,ollama_kwargs={},req=types.SimpleNamespace(stream_options={'include_usage':True}),_log_ollama_timings=lambda *a:None)
  exec(compile(ast.Module(body=[fix,node],type_ignores=[]),'exact-server-stream','exec'),ns)
  return [json.loads(x[6:]) for x in [v async for v in ns['stream']()] if x!='data: [DONE]\n\n']
 async def test_terminal_tool_call_preserved(self):
  chunks=[{'done':True,'message':{'tool_calls':[{'function':{'name':'terminal','arguments':{'command':'pwd'}}}]},'prompt_eval_count':2,'eval_count':3}]
  f=await self.frames(chunks)
  self.assertEqual(f[0]['choices'][0]['delta']['tool_calls'][0]['function']['name'],'terminal')
  self.assertEqual(json.loads(f[0]['choices'][0]['delta']['tool_calls'][0]['function']['arguments']),{'command':'pwd'})
  self.assertEqual(f[1]['choices'][0]['finish_reason'],'tool_calls')
 async def test_terminal_content_preserved(self):
  f=await self.frames([{'done':True,'message':{'content':'last'}}])
  self.assertEqual(f[0]['choices'][0]['delta'],{'content':'last'})
  self.assertEqual(f[1]['choices'][0]['finish_reason'],'stop')
 async def test_nonterminal_once(self):
  f=await self.frames([{'message':{'content':'first'}},{'done':True,'message':{}}])
  self.assertEqual([c['delta'].get('content') for x in f for c in x.get('choices',[]) if c['delta']],['first'])
 async def test_final_payload_follows_earlier_payload(self):
  f=await self.frames([{'message':{'content':'a'}},{'done':True,'message':{'content':'b'}}])
  self.assertEqual([x['choices'][0]['delta'].get('content') for x in f if x['choices'][0]['delta']],['a','b'])
  self.assertEqual(f[-1]['choices'][0]['finish_reason'],'stop')
 async def test_terminal_empty_not_fabricated(self):
  f=await self.frames([{'done':True,'message':{}}])
  self.assertEqual(len(f),1);self.assertEqual(f[0]['choices'][0]['delta'],{})
 async def test_both_content_and_tool(self):
  f=await self.frames([{'done':True,'message':{'content':'using tool','tool_calls':[{'function':{'name':'read_file','arguments':{'path':'a'}}}]}}])
  self.assertEqual(f[0]['choices'][0]['delta']['content'],'using tool');self.assertEqual(len(f),2)
unittest.main()
