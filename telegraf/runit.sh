# docker run -it --rm --name=telegraf \
#         -v $PWD/telegraf.conf:/etc/telegraf/telegraf.conf:ro \
#         -v $PWD:/wd \
#         -v /:/hostfs:ro \
#         -e HOST_ETC=/hostfs/etc \
#         -e HOST_PROC=/hostfs/proc \
#         -e HOST_SYS=/hostfs/sys \
#         -e HOST_VAR=/hostfs/var \
#         -e HOST_RUN=/hostfs/run \
#         -e HOST_MOUNT_PREFIX=/hostfs --network="host" \
#         telegraf


docker run -it --rm --name=telegraf \
        -v $PWD/telegraf.conf:/etc/telegraf/telegraf.conf:ro \
        encrypted_machines_crypten telegraf
