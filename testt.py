from zope.interface import implementer, Interface

from gen_server.base_types.architecture import IArchitecture


@implementer(IArchitecture)
class Testt:
    # def __init__(self, model=None, config=None):
    #     self.model = model
    #     self.config = config

    def display_name(self):
        pass

    def input_space(self):
        pass

    def output_space(self):
        pass

    def load(self, state_dict, device=None):
        pass

    def detect(self, state_dict):
        pass


if __name__ == "__main__":
    testt = Testt()
    #
    # try:
    #     print(verify.verifyObject(IArchitecture, testt))
    # except Invalid as e:
    #     logging.log(logging.WARNING, f"Error in verifying component IArchitecture {e}")
    # except Exception:
    #     logging.log("Unknownerror...")
    # print(IArchitecture.providedBy(testt))
    print(type(Interface))
