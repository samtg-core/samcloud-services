"""Pure configuration plans: no live profiles, credentials or native processes."""
from copy import deepcopy
import unittest

from .inference_native_profile import profile_plan,wrapper_plan


class NativePlan(unittest.TestCase):
    def config(self):
        return {'model':{'default':'qwen','provider':'custom','base_url':'https://queue.invalid/v1',
            'context_length':65536,'api_key':'${OPENAI_API_KEY}'},
            'compression':{'enabled':True,'threshold':.75},
            'terminal':{'cwd':'/work'},'agent':{'max_turns':100}}

    def test_preserves_threshold_context_tools_and_input_object(self):
        original=self.config();before=deepcopy(original)
        planned=profile_plan(original)
        self.assertEqual(original,before)
        self.assertEqual(planned['model']['context_length'],65536)
        self.assertEqual(planned['compression'],{'enabled':True,'threshold':.75,'context_timeout_seconds':0})
        self.assertEqual(planned['terminal'],before['terminal'])
        self.assertEqual(planned['agent'],before['agent'])
        self.assertEqual(planned['model']['provider'],'samcloud-queue-stint')
        self.assertEqual(planned['auxiliary']['compression']['provider'],'samcloud-queue-aux')
        self.assertNotIn('api_key',planned['auxiliary']['compression'])

    def test_preserves_explicit_token_cap_instead_of_introducing_a_new_one(self):
        config=self.config();config['compression']['threshold_tokens']=70000
        self.assertEqual(profile_plan(config)['compression']['threshold_tokens'],70000)
        self.assertNotIn('threshold_tokens',profile_plan(self.config())['compression'])

    def test_refuses_other_context_engine_or_plugin_isolation_change(self):
        for override in ({'context':{'engine':'other'}},
                         {'plugins':{'enabled':['other'],'isolation':'host'}}):
            with self.subTest(override=override),self.assertRaises(ValueError):
                profile_plan(self.config()|override)

    def test_existing_in_process_plugin_and_other_auxiliary_task_are_preserved(self):
        planned=profile_plan(self.config()|{'plugins':{'enabled':['other'],'isolation':'in_process'},
            'auxiliary':{'title':{'model':'title-model'}}})
        self.assertEqual(planned['plugins']['enabled'],['other','samcloud-queue-hooks'])
        self.assertEqual(planned['auxiliary']['title'],{'model':'title-model'})

    def test_different_summary_model_is_not_silently_moved_to_local_runner(self):
        with self.assertRaisesRegex(ValueError,'Different auxiliary model'):
            profile_plan(self.config()|{'auxiliary':{'compression':{'model':'other'}}})

    def test_existing_same_gateway_vision_keeps_its_model_inside_admission(self):
        vision={'provider':'custom','model':'vision-7b','base_url':'https://queue.invalid/v1',
                'api_key':'${OPENAI_API_KEY}'}
        planned=profile_plan(self.config()|{'auxiliary':{'vision':vision}})
        route=planned['auxiliary']['vision']
        self.assertEqual(route['provider'],'samcloud-queue-tool')
        self.assertEqual(route['model'],vision['model'])
        self.assertEqual(route['base_url'],vision['base_url'])
        self.assertNotIn('api_key',route)

    def test_external_vision_provider_is_not_migrated_by_the_queue_profile(self):
        vision={'provider':'custom','model':'vision-7b','base_url':'https://other.invalid/v1',
                'api_key':'${OTHER_API_KEY}'}
        planned=profile_plan(self.config()|{'auxiliary':{'vision':vision}})
        self.assertEqual(planned['auxiliary']['vision'],vision)

    def test_wrapper_binds_existing_credential_and_preserves_launch(self):
        source="import os\nos.environ['OPENAI_BASE_URL']='https://queue.invalid/v1'\nos.execv('/python',['/python','client.py'])\n"
        planned=wrapper_plan(source,'/absolute/source')
        environment={'OPENAI_API_KEY':'synthetic','PYTHONPATH':'/existing'}
        from types import SimpleNamespace
        calls=[]
        fake=SimpleNamespace(environ=environment,pathsep=':',execv=lambda *args:calls.append(args))
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules,{'os':fake}):exec(planned,{})
        self.assertEqual(environment['SAMCLOUD_QUEUE_API_KEY'],'synthetic')
        self.assertEqual(environment['SAMCLOUD_QUEUE_BASE_URL'],'https://queue.invalid/v1')
        self.assertEqual(environment['PYTHONPATH'],'/absolute/source:/existing')
        self.assertEqual(calls,[('/python',['/python','client.py'])])

    def test_wrapper_requires_known_boundary_and_refuses_double_install(self):
        for source in ('import os\nos.execvp("x",[])\n',
                       'import os\nos.execv("x",[])\nprint("after")\n',
                       'import os\nSAMCLOUD_QUEUE_API_KEY="x"\nos.execv("x",[])\n'):
            with self.subTest(source=source),self.assertRaises(ValueError):
                wrapper_plan(source,'/absolute/source')

    def test_guard_blocks_missing_partial_or_drifted_activation_before_native_exec(self):
        import hashlib,json,sys,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'activation.json';config=root/'config.yaml'
            wrapper=root/'wrapper.py'
            (root/'source').mkdir();module=root/'source/module.py';module.write_text('# original\n')
            source="import os\nos.environ['OPENAI_BASE_URL']='https://queue.invalid/v1'\nos.execv('/python',['/python'])\n"
            planned=wrapper_plan(source,root/'source',activation_manifest=manifest)
            wrapper.write_text(planned);config.write_text('original')
            calls=[]
            fake=SimpleNamespace(environ={'OPENAI_API_KEY':'synthetic'},pathsep=':',
                                 execv=lambda *args:calls.append(args))
            active={'state':'active','source_path':str(root/'source'),'config_path':str(config),
                'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
                'source_files':{'module.py':hashlib.sha256(module.read_bytes()).hexdigest()},
                'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest()}
            def run():
                with patch.dict(sys.modules,{'os':fake}):
                    exec(planned,{'__file__':str(wrapper)})
            with self.assertRaisesRegex(SystemExit,'no native client started'):run()
            manifest.write_text(json.dumps(active|{'state':'preparing'}))
            with self.assertRaises(SystemExit):run()
            manifest.write_text(json.dumps(active));config.write_text('drifted')
            with self.assertRaises(SystemExit):run()
            self.assertEqual(calls,[])
            config.write_text('original');run()
            self.assertEqual(calls,[('/python',['/python'])])
            module.write_text('# drift\n')
            with self.assertRaises(SystemExit):run()
            module.write_text('# original\n')
            wrapper.write_text(planned+'# changed\n')
            with self.assertRaises(SystemExit):run()
            self.assertEqual(len(calls),1)


if __name__ == '__main__':unittest.main()
