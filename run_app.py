from huahua_assistant.capture import enable_process_dpi_awareness


enable_process_dpi_awareness()


from huahua_assistant.app import HuahuaApp, main
from huahua_assistant.desktop_pet import install_desktop_pet_markers


install_desktop_pet_markers(HuahuaApp)


if __name__ == "__main__":
    main()
