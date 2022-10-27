import os
import unittest
from typing import Optional, List

from commonsnakes import load_from_envs, RequiredSetting
from commonsnakes.env_loader import __env_var_name_of_attr as get_env_name

def clear_test_env_variables():
    envs = os.environ
    custom_settings: List[str] = [key for key in envs.keys() if "TEST_TESTSETTINGS" in key]
    for custom_setting in custom_settings:
        del envs[custom_setting]

class EnvLoaderTestCase(unittest.TestCase):


    def tearDown(self) -> None:
        super().tearDown()
        clear_test_env_variables()

    def load_settings_class(self):
        @load_from_envs("TEST_")
        class TestSettings:
            str1 = "DefaultX1"
            str2: str = "DefaultX2"
            str3: Optional[str] = None

            int1 = 3
            int2: int = 4
            int3: Optional[int] = None

            b1: Optional[bool] = None
            b2: bool = False

            f1: Optional[float] = None
            f2: float = 3.14159

            non_existing: str

        return TestSettings

    def test_loading_defaults(self):
        t = self.load_settings_class()
        self.assertEqual(t.str1, "DefaultX1")
        self.assertEqual(t.str2, "DefaultX2")
        self.assertEqual(t.str3, None)

        self.assertEqual(t.int1, 3)
        self.assertEqual(t.int2, 4)
        self.assertEqual(t.int3, None)

        self.assertEqual(t.b1, None)
        self.assertEqual(t.b2, False)

        self.assertEqual(t.f1, None)
        self.assertEqual(t.f2, 3.14159)

    def test_non_existing(self):
        t = self.load_settings_class()
        self.failUnlessRaises(AttributeError, lambda: print(t.non_existing))

    def test_non_existing_loaded(self):
        # t = self.load_class()
        envs = os.environ
        env_name = 'TEST_TESTSETTINGS_NON_EXISTING' # get_env_name(t, "non_existing", "TEST_")
        envs[env_name] = "Hello"
        t = self.load_settings_class()
        self.failUnlessRaises(AttributeError, lambda: print(t.non_existing))

    def test_loading(self):
        envs = os.environ
        env_name = 'TEST_TESTSETTINGS_STR1'
        envs[env_name] = "Hello1"
        env_name = 'TEST_TESTSETTINGS_STR2'
        envs[env_name] = "Hello2"
        env_name = 'TEST_TESTSETTINGS_STR3'
        envs[env_name] = "Hello3"
        t = self.load_settings_class()
        self.assertEqual(t.str1, "Hello1")
        self.assertEqual(t.str2, "Hello2")
        self.assertEqual(t.str3, "Hello3")

        env_name = 'TEST_TESTSETTINGS_INT1'
        envs[env_name] = "1"
        env_name = 'TEST_TESTSETTINGS_INT2'
        envs[env_name] = "2"
        env_name = 'TEST_TESTSETTINGS_INT3'
        envs[env_name] = "3"
        t = self.load_settings_class()
        self.assertEqual(t.int1, "1")
        self.assertEqual(t.int2, 2)
        self.assertEqual(t.int3, 3)

        env_name = 'TEST_TESTSETTINGS_B1'
        envs[env_name] = ""
        env_name = 'TEST_TESTSETTINGS_B2'
        envs[env_name] = "1"
        t = self.load_settings_class()
        self.assertEqual(t.b1, False)
        self.assertEqual(t.b2, True)

        env_name = 'TEST_TESTSETTINGS_F1'
        envs[env_name] = "1.5"
        env_name = 'TEST_TESTSETTINGS_F2'
        envs[env_name] = "3"
        t = self.load_settings_class()
        self.assertEqual(t.f1, 1.5)
        self.assertEqual(t.f2, 3.0)


class RequiredSettingTestCase(unittest.TestCase):

    def tearDown(self) -> None:
        super().tearDown()
        clear_test_env_variables()

    def load_settings_class(self):
        @load_from_envs("Test_")
        class TestSettings:

            a: str = None
            b: str = RequiredSetting()
            c: int = RequiredSetting("Please provide c")

        return TestSettings

    def test_require_setting_not_set(self):
        t = self.load_settings_class()
        self.assertEqual(t.a, None)

        error_thrown = False
        try:
            print(t.b)
        except RuntimeError as e:
            error_thrown = True
            self.assertIn('TEST_TESTSETTINGS_B', str(e))
        self.assertTrue(error_thrown)

        error_thrown = False
        try:
            print(t.c)
        except RuntimeError as e:
            error_thrown = True
            self.assertIn('TEST_TESTSETTINGS_C', str(e))
            self.assertIn('Please provide c', str(e))

        self.assertTrue(error_thrown)

    def test_required_setting_set(self):
        os.environ["TEST_TESTSETTINGS_A"] = "Hello"
        os.environ["TEST_TESTSETTINGS_B"] = "World"
        os.environ["TEST_TESTSETTINGS_C"] = "10"
        t = self.load_settings_class()
        self.assertEqual(t.a, "Hello")
        self.assertEqual(t.b, "World")
        self.assertEqual(t.c, 10)

    def test_fail_fast(self):
        def load_fail_fast_setting():
            @load_from_envs("Test_")
            class TestSettings:
                b: str = RequiredSetting()
                c: int = RequiredSetting(fail_fast=True)
            return TestSettings
        self.assertRaises(RuntimeError, lambda: load_fail_fast_setting())
        os.environ["TEST_TESTSETTINGS_C"] = "10"
        t = load_fail_fast_setting()
        self.assertEqual(t.c, 10)
        self.failUnlessRaises(RuntimeError, lambda: print(t.b))


if __name__ == '__main__':
    unittest.main()
